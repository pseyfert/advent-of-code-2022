use anyhow::Result;
use itertools::fold;

pub fn get_best<I>(x: I) -> Result<i32>
where
    I: Iterator<Item = Option<i32>>,
{
    let (_, best) = fold(x, (0, 0), |(cur_sum, max), iter| match iter {
        None => (0, i32::max(cur_sum, max)),
        Some(i) => (cur_sum + i, max),
    });
    Ok(best)
}
