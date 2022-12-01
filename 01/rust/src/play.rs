use anyhow::Result;
use itertools::Itertools;
use std::fs::File;
use std::io;
use std::io::BufRead;
use std::path::Path;

pub mod part2;

fn count_calories<P>(p: P) -> Result<()>
where
    P: AsRef<Path>,
{
    let file = File::open(p)?;
    let reader = io::BufReader::new(file);

    let result = reader
        .lines()
        .filter_map(|s| s.ok())
        .group_by(|s| !s.is_empty())
        .into_iter()
        .filter_map(|(key, group)| {
            if key {
                group.filter_map(|s| s.parse::<i32>().ok()).sum1()
            } else {
                None
            }
        })
        .fold([0, 0, 0], |top3, iter| part2::three_of_four(iter, top3));

    println!("part1: {}", result[0]);
    println!("part2: {}", result.iter().sum::<i32>());

    Ok(())
}

fn main() -> anyhow::Result<()> {
    count_calories("../input.txt")
}
