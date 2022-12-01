use anyhow::{Context, Result};
use std::fs::File;
use std::io;
use std::io::BufRead;
use std::iter::Map;

// use crate::part1;
use crate::part2;

// pub fn read_lines<P>(filename: P) -> Result<Map<io::Lines<io::BufReader<File>>, dyn FnMut(Result<String, std::io::Error>,)>>
// where
//     P: AsRef<std::path::Path>,
pub fn read_lines<P>(filename: P) -> Result<i32>
where
    P: AsRef<std::path::Path>,
{
    let file = File::open(filename)?;
    let reader = io::BufReader::new(file);

    let x = reader
        .lines()
        .into_iter()
        .map(|x: io::Result<String>| -> Option<i32> {
            x.context("error reading line")
                .and_then(|x| {
                    if x.is_empty() {
                        Ok(None)
                    } else {
                        x.parse::<i32>()
                            .and_then(|x| Ok(Some(x)))
                            .context("error parsing string -> i32")
                    }
                })
                .ok()?
        });
    // would love to return x here

    part2::get_res(x)
    // part1::get_best(x)
}
