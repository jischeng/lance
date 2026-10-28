// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Expose separated membership predicates to DataFusion's set simplification rules.

use std::collections::HashMap;

use datafusion::common::tree_node::{Transformed, TreeNode};
use datafusion::common::{Column, Result};
use datafusion::logical_expr::expr::InList;
use datafusion::logical_expr::{BinaryExpr, Expr, Operator};
use datafusion::scalar::ScalarValue;

pub fn simplify_membership_regions(
    expr: Expr,
    simplify: &impl Fn(Expr) -> Result<Expr>,
) -> Result<Expr> {
    let operator = match &expr {
        Expr::BinaryExpr(binary) if matches!(binary.op, Operator::And | Operator::Or) => binary.op,
        _ => {
            return expr
                .map_children(|child| {
                    simplify_membership_regions(child, simplify).map(Transformed::yes)
                })
                .map(|result| result.data);
        }
    };

    // Visit a whole homogeneous region once. Postorder shape markers let us
    // reconstruct its original parentheses without recursive descent through a
    // potentially long AND/OR chain. Only boundary expressions recurse.
    let mut pending = vec![Some(expr)];
    let mut shape = Vec::new();
    let mut terms = Vec::new();
    while let Some(item) = pending.pop() {
        match item {
            Some(Expr::BinaryExpr(binary)) if binary.op == operator => {
                pending.push(None);
                pending.push(Some(*binary.right));
                pending.push(Some(*binary.left));
            }
            Some(term) => {
                terms.push(Some(simplify_membership_regions(term, simplify)?));
                shape.push(false);
            }
            None => shape.push(true),
        }
    }

    // Store positions rather than Expr keys: column hashing is cheap and has no
    // interior mutability. Candidates never replace original terms until merged.
    let mut groups: HashMap<Column, Vec<(usize, InList)>> = HashMap::new();
    for (index, term) in terms.iter().enumerate() {
        let mut list = match term.as_ref() {
            Some(Expr::InList(list)) => list.clone(),
            Some(Expr::BinaryExpr(binary))
                if matches!(binary.op, Operator::Eq | Operator::NotEq) =>
            {
                let ((column @ Expr::Column(_), value) | (value, column @ Expr::Column(_))) =
                    (binary.left.as_ref(), binary.right.as_ref())
                else {
                    groups.clear();
                    break;
                };
                InList::new(
                    Box::new(column.clone()),
                    vec![value.clone()],
                    binary.op == Operator::NotEq,
                )
            }
            _ => continue,
        };
        let Expr::Column(column) = list.expr.as_ref() else {
            groups.clear();
            break;
        };
        if list.list.is_empty() {
            groups.clear();
            break;
        }
        // Coercion can insert casts around constants. Fold operands without
        // letting the simplifier expand the membership predicate itself.
        for value in &mut list.list {
            if !matches!(value, Expr::Literal(_, _)) {
                *value = simplify(value.clone())?;
            }
        }
        // Ineligible memberships must retain their surrounding terms too:
        // merging another column could make them adjacent and expose them to
        // structural set operations that do not model NULLs or float equality.
        if !list.list.iter().all(|item| {
            matches!(item,
            Expr::Literal(value, _) if !value.is_null() && !value.data_type().is_floating())
        }) {
            groups.clear();
            break;
        }
        groups
            .entry(column.clone())
            .or_default()
            .push((index, list));
    }
    // Removing another column's terms can expose a skipped group's predicates
    // to unsafe folding in the subsequent simplifier pass. Stage all changes
    // so a safety fallback preserves the entire region, regardless of group order.
    let mut replacements = Vec::with_capacity(groups.len());
    for candidates in groups.into_values() {
        if candidates.len() < 2 {
            continue;
        }
        let has_positive = candidates.iter().any(|(_, list)| !list.negated);
        let has_negative = candidates.iter().any(|(_, list)| list.negated);
        // DataFusion can lose UNKNOWN when folding complementary OR predicates.
        if operator == Operator::Or && has_positive && has_negative {
            replacements.clear();
            break;
        }
        let Some((last, _)) = candidates.last() else {
            continue;
        };
        let last = *last;
        // The simplifier recursively traverses its input. Keep the candidate
        // tree balanced even when the original region contains many terms.
        let mut level: Vec<_> = candidates
            .iter()
            .map(|(_, list)| Expr::InList(list.clone()))
            .collect();
        while level.len() > 1 {
            let mut parents = Vec::with_capacity(level.len().div_ceil(2));
            let mut children = level.into_iter();
            while let Some(left) = children.next() {
                parents.push(match children.next() {
                    Some(right) => {
                        Expr::BinaryExpr(BinaryExpr::new(Box::new(left), operator, Box::new(right)))
                    }
                    None => left,
                });
            }
            level = parents;
        }
        let Some(group) = level.pop() else { continue };
        let simplified = simplify(group)?;
        // A constant result loses UNKNOWN for a NULL column. A CASE guard
        // preserves that result but introduces residual filtering, which can
        // disable an enclosing OR's scalar index plan. Keep the whole region
        // so both three-valued logic and index eligibility are preserved.
        if matches!(&simplified, Expr::Literal(ScalarValue::Boolean(Some(_)), _)) {
            replacements.clear();
            break;
        }
        replacements.push((candidates, last, simplified));
    }
    for (candidates, last, simplified) in replacements {
        for (index, _) in candidates {
            terms[index] = None;
        }
        terms[last] = Some(simplified);
    }

    let mut terms = terms.into_iter();
    let mut rebuilt = Vec::new();
    for is_binary in shape {
        if !is_binary {
            rebuilt.push(terms.next().flatten());
            continue;
        }
        let right = rebuilt.pop().flatten();
        let left = rebuilt.pop().flatten();
        rebuilt.push(match (left, right) {
            (Some(left), Some(right)) => Some(Expr::BinaryExpr(BinaryExpr::new(
                Box::new(left),
                operator,
                Box::new(right),
            ))),
            (remaining, None) | (None, remaining) => remaining,
        });
    }
    rebuilt.pop().flatten().ok_or_else(|| {
        datafusion::common::internal_datafusion_err!(
            "Membership simplification removed every region term"
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::logical_expr::{col, lit};
    use std::cell::Cell;

    #[test]
    fn preserves_unmerged_regions_without_simplifier_calls() {
        let mut level: Vec<_> = (0..1024).map(|i| col(format!("c{i}")).eq(lit(i))).collect();
        while level.len() > 1 {
            level = level
                .chunks(2)
                .map(|pair| pair[0].clone().and(pair[1].clone()))
                .collect();
        }
        let original = level.pop().unwrap();
        let calls = Cell::new(0);
        let rewritten = simplify_membership_regions(original.clone(), &|expr| {
            calls.set(calls.get() + 1);
            Ok(expr)
        })
        .unwrap();
        assert_eq!(rewritten, original);
        assert_eq!(calls.get(), 0);
    }

    #[test]
    fn preserves_mixed_or_candidates() {
        let original = col("s")
            .eq(lit("a"))
            .or(col("i").eq(lit(1)))
            .or(col("s").not_eq(lit("b")))
            .or(col("i").eq(lit(1)));
        let rewritten =
            simplify_membership_regions(original.clone(), &|_| Ok(col("i").eq(lit(1)))).unwrap();
        assert_eq!(rewritten, original);
    }

    #[rstest::rstest]
    #[case::empty_intersection(Operator::And, false)]
    #[case::empty_negated_intersection(Operator::Or, true)]
    fn preserves_regions_when_simplification_is_constant(
        #[case] operator: Operator,
        #[case] negated: bool,
    ) {
        let first = col("s").in_list(vec![lit("a"), lit("b")], negated);
        let middle = col("i").eq(lit(1));
        let last = col("s").in_list(vec![lit("c"), lit("d")], negated);
        let original = Expr::BinaryExpr(BinaryExpr::new(
            Box::new(first),
            operator,
            Box::new(Expr::BinaryExpr(BinaryExpr::new(
                Box::new(middle),
                operator,
                Box::new(last),
            ))),
        ));
        let calls = Cell::new(0);
        let rewritten = simplify_membership_regions(original.clone(), &|_| {
            calls.set(calls.get() + 1);
            Ok(lit(negated))
        })
        .unwrap();
        assert_eq!(rewritten, original);
        assert_eq!(calls.get(), 1);
    }

    #[test]
    fn discards_staged_merges_when_another_group_needs_fallback() {
        let original = col("s")
            .eq(lit("a"))
            .and(col("t").eq(lit("a")))
            .and(col("s").not_eq(lit("a")))
            .and(col("t").eq(lit("a")));
        let calls = Cell::new(0);
        let rewritten = simplify_membership_regions(original.clone(), &|expr| {
            calls.set(calls.get() + 1);
            // Force a successful merge before the unsafe result, independent
            // of HashMap iteration order.
            if calls.get() == 1 {
                Ok(expr)
            } else {
                Ok(lit(false))
            }
        })
        .unwrap();
        assert_eq!(calls.get(), 2);
        assert_eq!(rewritten, original);
    }

    #[test]
    fn simplifies_each_region_once() {
        let original = col("s")
            .in_list(vec![lit("a"), lit("b")], false)
            .and(col("i").eq(lit(1)))
            .and(col("s").in_list(vec![lit("b"), lit("c")], false));
        let calls = Cell::new(0);
        // Returning the group unchanged deliberately simulates a simplifier
        // that cannot improve a candidate. The output must not be revisited.
        simplify_membership_regions(original, &|expr| {
            calls.set(calls.get() + 1);
            Ok(expr)
        })
        .unwrap();
        assert_eq!(calls.get(), 1);
    }
}
