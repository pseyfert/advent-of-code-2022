use anyhow::Result;
use itertools::{fold, sorted};
use std::convert::TryInto;

fn three_of_four(x: i32, y: [i32; 3]) -> [i32; 3] {
    // Is this sorted actually lazy?
    // For the search of TopN elements one "only" needs to buffer N elements when iterating.
    // Wondering if that's done here?
    sorted(y.iter().chain(&[x]))
        .rev()
        .take(3)
        .map(|x| *x)
        .collect::<Vec<i32>>()
        .try_into()
        .unwrap()
}

fn get_top_3<I>(x: I) -> Result<[i32; 3]>
where
    I: Iterator<Item = Option<i32>>,
{
    let (_, best) = fold(x, (0, [0, 0, 0]), |(cur_sum, top3), iter| match iter {
        None => (0, three_of_four(cur_sum, top3)),
        Some(i) => (cur_sum + i, top3),
    });
    Ok(best)
}

pub fn get_res<I>(x: I) -> Result<i32>
where
    I: Iterator<Item = Option<i32>>,
{
    let t = get_top_3(x)?;
    Ok(t.iter().sum())
}
