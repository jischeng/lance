// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Expose separated membership predicates to DataFusion's set simplification rules.

use std::collections::HashMap;
use std::mem;

use datafusion::common::tree_node::{Transformed, TreeNode};
use datafusion::common::{Column, Result};
use datafusion::logical_expr::expr::InList;
use datafusion::logical_expr::{BinaryExpr, Expr, Operator};
use datafusion::scalar::ScalarValue;

/// A borrowed membership, including equality and explicit NOT spellings.
struct Membership<'a> {
    expr: &'a Expr,
    values: &'a [Expr],
    negated: bool,
}

enum Term<'a> {
    Membership(Membership<'a>),
    /// Negating a compound predicate can expose memberships after this pass.
    Barrier,
    Other,
}

/// Group separated memberships within each AND/OR region before simplification.
///
/// Replacements are applied together: moving one column's predicates must not
/// expose an unsafe group on another column to DataFusion's set algebra.
pub fn simplify_membership_regions(
    mut expr: Expr,
    simplify: &impl Fn(Expr) -> Result<Expr>,
) -> Result<Expr> {
    rewrite(&mut expr, simplify)?;
    Ok(expr)
}

fn rewrite(expr: &mut Expr, simplify: &impl Fn(Expr) -> Result<Expr>) -> Result<()> {
    match expr {
        Expr::BinaryExpr(binary) if matches!(binary.op, Operator::And | Operator::Or) => {
            let operator = binary.op;
            // Visit only this region. Boundary expressions recurse independently,
            // including regions inside NOT; no ancestor rescans those subtrees.
            let mut pending = vec![&mut *expr];
            let mut terms = Vec::new();
            let mut shape = Vec::new();
            while let Some(term) = pending.pop() {
                if matches!(term, Expr::BinaryExpr(binary) if binary.op == operator) {
                    shape.push(true);
                    if let Expr::BinaryExpr(binary) = term {
                        pending.push(binary.right.as_mut());
                        pending.push(binary.left.as_mut());
                    }
                } else {
                    shape.push(false);
                    rewrite(term, simplify)?;
                    terms.push(term);
                }
            }
            let replacements = plan_replacements(&terms, operator, simplify)?;
            if replacements.is_empty() {
                return Ok(());
            }
            let mut removed = vec![false; terms.len()];
            let mut merged: Vec<_> = (0..terms.len()).map(|_| None).collect();
            for (indices, replacement) in replacements {
                let last = indices[indices.len() - 1];
                for index in indices {
                    removed[index] = true;
                }
                removed[last] = false;
                merged[last] = Some(replacement);
            }
            rebuild_region(expr, operator, &shape, &removed, &mut merged)?;
        }
        Expr::BinaryExpr(binary) => {
            rewrite(&mut binary.left, simplify)?;
            rewrite(&mut binary.right, simplify)?;
        }
        Expr::InList(list) => {
            rewrite(&mut list.expr, simplify)?;
            for value in &mut list.list {
                rewrite(value, simplify)?;
            }
        }
        Expr::Column(_) | Expr::Literal(_, _) => {}
        _ => {
            let original = mem::replace(expr, Expr::Literal(ScalarValue::Null, None));
            *expr = original
                .map_children(|mut child| {
                    rewrite(&mut child, simplify)?;
                    Ok(Transformed::yes(child))
                })?
                .data;
        }
    }
    Ok(())
}

fn plan_replacements(
    terms: &[&mut Expr],
    operator: Operator,
    simplify: &impl Fn(Expr) -> Result<Expr>,
) -> Result<Vec<(Vec<usize>, Expr)>> {
    // Column lookup is constant-time. Computed expressions are uncommon and
    // compared directly, without hashing Expr values that can contain UDFs.
    let mut columns: HashMap<&Column, usize> = HashMap::new();
    let mut groups: Vec<Vec<(usize, Membership<'_>)>> = Vec::new();
    let mut computed_groups = Vec::new();
    for (index, term) in terms.iter().enumerate() {
        let membership = match classify_term(term) {
            Term::Membership(membership) => membership,
            Term::Barrier => return Ok(Vec::new()),
            Term::Other => continue,
        };
        let group_index = if let Expr::Column(column) = membership.expr {
            *columns.entry(column).or_insert_with(|| {
                groups.push(Vec::new());
                groups.len() - 1
            })
        } else if let Some(&group_index) = computed_groups.iter().find(|&&group_index| {
            let group: &Vec<(usize, Membership<'_>)> = &groups[group_index];
            group[0].1.expr == membership.expr
        }) {
            group_index
        } else {
            let group_index = groups.len();
            computed_groups.push(group_index);
            groups.push(Vec::new());
            group_index
        };
        groups[group_index].push((index, membership));
    }

    let mut replacements = Vec::with_capacity(groups.len());
    for group in groups {
        if group.len() < 2 {
            continue;
        }
        let has_positive = group.iter().any(|(_, membership)| !membership.negated);
        let has_negative = group.iter().any(|(_, membership)| membership.negated);
        if !matches!(group[0].1.expr, Expr::Column(_))
            || (operator == Operator::Or && has_positive && has_negative)
        {
            return Ok(Vec::new());
        }
        let mut indices = Vec::with_capacity(group.len());
        let mut level = Vec::with_capacity(group.len());
        for (index, membership) in group {
            if membership.values.is_empty() {
                return Ok(Vec::new());
            }
            // Clone operands only after a repeated column is found. Fold casts
            // on constants without expanding the membership predicate itself.
            let values = membership
                .values
                .iter()
                .map(|value| match value {
                    Expr::Literal(_, _) => Ok(value.clone()),
                    _ => simplify(value.clone()),
                })
                .collect::<Result<Vec<_>>>()?;
            // Float normalization may introduce NaN and signed-zero alternatives.
            // Do not reduce them using structural literal equality. NULL list
            // elements also do not obey the ordinary set-algebra rules.
            if !values.iter().all(|value| {
                matches!(value,
                    Expr::Literal(value, _) if !value.is_null() && !value.data_type().is_floating()
                )
            }) {
                return Ok(Vec::new());
            }
            indices.push(index);
            level.push(Expr::InList(InList::new(
                Box::new(membership.expr.clone()),
                values,
                membership.negated,
            )));
        }
        // Keep the input to the recursive DataFusion simplifier logarithmic in
        // the number of predicates, even for a long original AND/OR chain.
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
        let simplified = simplify(level.remove(0))?;
        // A constant drops UNKNOWN. A CASE guard introduces residual filtering
        // and can disable an enclosing OR's index plan. Preserve the whole
        // region instead, including separators belonging to other columns.
        if matches!(simplified, Expr::Literal(ScalarValue::Boolean(_), _)) {
            return Ok(Vec::new());
        }
        replacements.push((indices, simplified));
    }
    Ok(replacements)
}

fn classify_term(mut term: &Expr) -> Term<'_> {
    let mut negated = false;
    while let Expr::Not(inner) = term {
        negated = !negated;
        term = inner;
    }
    match term {
        Expr::InList(list) => Term::Membership(Membership {
            expr: &list.expr,
            values: &list.list,
            negated: negated ^ list.negated,
        }),
        Expr::BinaryExpr(binary) if matches!(binary.op, Operator::Eq | Operator::NotEq) => {
            let (expr, value) = match (binary.left.as_ref(), binary.right.as_ref()) {
                (Expr::Column(_), Expr::Column(_)) => return Term::Other,
                (expr, value @ Expr::Literal(_, _))
                | (value @ Expr::Literal(_, _), expr)
                | (expr @ Expr::Column(_), value)
                | (value, expr @ Expr::Column(_)) => (expr, value),
                _ => return Term::Barrier,
            };
            Term::Membership(Membership {
                expr,
                values: std::slice::from_ref(value),
                negated: negated ^ (binary.op == Operator::NotEq),
            })
        }
        Expr::BinaryExpr(binary)
            if negated && matches!(binary.op, Operator::And | Operator::Or) =>
        {
            Term::Barrier
        }
        // NOT bool and NOT (IS NULL) cannot become a same-column membership.
        Expr::Column(_) | Expr::IsNull(_) | Expr::IsNotNull(_) => Term::Other,
        _ if negated => Term::Barrier,
        _ => Term::Other,
    }
}

/// Prune replaced terms without changing the parenthesization of those kept.
fn rebuild_region(
    expr: &mut Expr,
    operator: Operator,
    shape: &[bool],
    removed: &[bool],
    merged: &mut [Option<Expr>],
) -> Result<()> {
    let original = mem::replace(expr, Expr::Literal(ScalarValue::Null, None));
    let mut pending = vec![Some(original)];
    let mut rebuilt = Vec::new();
    let mut index = 0;
    let mut nodes = shape.iter();
    while let Some(item) = pending.pop() {
        match item {
            Some(term) => {
                // A rewritten boundary can now have this region's operator.
                // Follow the original shape so it still occupies one term.
                if nodes.next() == Some(&true) {
                    let Expr::BinaryExpr(binary) = term else {
                        return Err(datafusion::common::internal_datafusion_err!(
                            "Membership region lost a binary node"
                        ));
                    };
                    pending.push(None);
                    pending.push(Some(*binary.right));
                    pending.push(Some(*binary.left));
                } else {
                    rebuilt.push(if removed[index] {
                        None
                    } else {
                        Some(merged[index].take().unwrap_or(term))
                    });
                    index += 1;
                }
            }
            None => {
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
        }
    }
    *expr = rebuilt.pop().flatten().ok_or_else(|| {
        datafusion::common::internal_datafusion_err!(
            "Membership simplification removed every region term"
        )
    })?;
    Ok(())
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
    fn preserves_allocations_when_no_column_repeats() {
        let original = col("s")
            .in_list((0..100_000).map(lit).collect(), false)
            .and(col("i").lt(lit(1)));
        let Expr::BinaryExpr(binary) = &original else {
            unreachable!()
        };
        let left = binary.left.as_ref() as *const Expr;
        let Expr::InList(list) = binary.left.as_ref() else {
            unreachable!()
        };
        let values = list.list.as_ptr();
        let rewritten = simplify_membership_regions(original, &|_| {
            panic!("A single membership must not invoke the simplifier")
        })
        .unwrap();
        let Expr::BinaryExpr(binary) = rewritten else {
            unreachable!()
        };
        assert_eq!(binary.left.as_ref() as *const Expr, left);
        let Expr::InList(list) = binary.left.as_ref() else {
            unreachable!()
        };
        assert_eq!(list.list.as_ptr(), values);
        assert_eq!(list.list.len(), 100_000);
    }

    #[rstest::rstest]
    #[case::float(col("f"), vec![lit(1.0), lit(2.0)])]
    #[case::null_element(col("f"), vec![lit(ScalarValue::Int32(None))])]
    #[case::computed(col("f") + lit(1), vec![lit(1), lit(2)])]
    fn preserves_repeated_ineligible_memberships(#[case] expr: Expr, #[case] values: Vec<Expr>) {
        let original = expr
            .clone()
            .in_list(values.clone(), false)
            .and(col("s").eq(lit("a")))
            .and(expr.in_list(values, true))
            .and(col("s").eq(lit("a")));
        let rewritten = simplify_membership_regions(original.clone(), &|_| {
            panic!("Repeated ineligible memberships must preserve the region")
        })
        .unwrap();
        assert_eq!(rewritten, original);
    }

    #[test]
    fn preserves_mixed_or_candidates() {
        let original = col("s")
            .eq(lit("a"))
            .or(col("i").eq(lit(1)))
            .or(col("s").not_eq(lit("b")))
            .or(col("i").eq(lit(1)));
        let rewritten = simplify_membership_regions(original.clone(), &|_| {
            panic!("Mixed-polarity OR must reject the region before simplifying a group")
        })
        .unwrap();
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
            // Simulate a successful group followed by an unsafe constant.
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
    fn merges_memberships_around_a_column_comparison() {
        let original = col("s")
            .in_list(vec![lit("a"), lit("b")], false)
            .and(col("t").eq(col("u")))
            .and(col("s").in_list(vec![lit("b"), lit("c")], false));
        let calls = Cell::new(0);
        let rewritten = simplify_membership_regions(original.clone(), &|expr| {
            calls.set(calls.get() + 1);
            Ok(expr)
        })
        .unwrap();
        assert_eq!(calls.get(), 1);
        assert_ne!(rewritten, original);
    }

    #[test]
    fn simplifies_each_region_once() {
        let original = col("s")
            .in_list(vec![lit("a"), lit("b")], false)
            .and(col("i").eq(lit(1)))
            .and(col("s").in_list(vec![lit("b"), lit("c")], false));
        let calls = Cell::new(0);
        // A non-improving simplifier must not cause the output to be revisited.
        simplify_membership_regions(original, &|expr| {
            calls.set(calls.get() + 1);
            Ok(expr)
        })
        .unwrap();
        assert_eq!(calls.get(), 1);
    }
}
