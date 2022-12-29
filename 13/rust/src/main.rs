use anyhow::{anyhow, bail, Context, Result};
use env_logger;
use itertools::EitherOrBoth::{Both, Left, Right};
use itertools::Itertools;
use log::{debug, info};
use rayon::prelude::*;
use std::cmp::Ordering;
use std::fs::File;
use std::io;
use std::io::BufRead;
use std::path::Path;

#[derive(Debug, Clone)]
enum Packet {
    Val(i32),
    List(Vec<Packet>),
}

impl PartialOrd for Packet {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        debug!("Comparing {:?} to {:?}", &self, &other);
        match self {
            Packet::Val(lhs_int) => match other {
                Packet::Val(rhs_int) => lhs_int.partial_cmp(rhs_int),
                Packet::List(rhs_list) => {
                    Packet::List(vec![Packet::Val(*lhs_int)]).partial_cmp(other)
                }
            },
            Packet::List(lhs_list) => match other {
                Packet::Val(rhs_int) => {
                    self.partial_cmp(&Packet::List(vec![Packet::Val(*rhs_int)]))
                }
                Packet::List(rhs_list) => {
                    for zip_iter in lhs_list.into_iter().zip_longest(rhs_list) {
                        match zip_iter {
                            Right(_) => {
                                debug!("LHS ran out first");
                                return Some(Ordering::Less);
                            }
                            Left(_) => {
                                debug!("RHS ran out first");
                                return Some(Ordering::Greater);
                            }
                            Both(l, r) => match l.partial_cmp(r) {
                                None | Some(Ordering::Equal) => {
                                    debug!("go to next element in list");
                                }
                                Some(d) => {
                                    return Some(d);
                                }
                            },
                        }
                    }
                    debug!("Lists exhausted, go to next");
                    return None;
                }
            },
        }
    }
}

// impl PartialEq for Packet {
//     fn eq(&self, other: &Self) -> bool {
//         self.partial_cmp(other) == Some(Ordering::Equal)
//     }
// }

impl PartialEq for Packet {
    fn eq(&self, other: &Self) -> bool {
        match self {
            Packet::Val(i) => match other {
                Packet::Val(o) => i == o,
                _ => false,
            },
            Packet::List(tl) => match other {
                Packet::List(ol) => tl
                    .into_iter()
                    .zip_longest(ol)
                    .all(|zip_iter| match zip_iter {
                        Left(_) | Right(_) => false,
                        Both(lhs_e, rhs_e) => rhs_e == lhs_e,
                    }),
                _ => false,
            },
        }
    }
}

impl Packet {
    fn from_str(s: &str) -> Option<Packet> {
        if s.is_empty() {
            None
        } else {
            Some({
                let (first, mut to_be_packed) = s.split_at(1);
                match first {
                    "[" => {
                        // TODO
                        let mut retval_inner: Vec<Packet> = vec![];
                        let mut parenthesis_stack = 0;
                        'outer: while !to_be_packed.is_empty() {
                            for (i, c) in to_be_packed.chars().enumerate() {
                                match c {
                                    '[' => parenthesis_stack += 1,
                                    ']' => {
                                        if parenthesis_stack == 0 {
                                            let pack;
                                            (pack, to_be_packed) = to_be_packed.split_at(i);
                                            (_, to_be_packed) = to_be_packed.split_at(1);
                                            // debug!("sending '{}' to recursion due to ]", pack);
                                            // debug!("keeping '{}' on the stack", to_be_packed);
                                            if let Some(pack) = Packet::from_str(pack) {
                                                retval_inner.push(pack);
                                            }
                                            continue 'outer;
                                        } else {
                                            parenthesis_stack -= 1;
                                        }
                                    }
                                    ',' => {
                                        if parenthesis_stack == 0 {
                                            let pack;
                                            (pack, to_be_packed) = to_be_packed.split_at(i);
                                            (_, to_be_packed) = to_be_packed.split_at(1);
                                            // debug!("sending '{}' to recursion due to ,", pack);
                                            // debug!("keeping '{}' on the stack", to_be_packed);
                                            if let Some(pack) = Packet::from_str(pack) {
                                                retval_inner.push(pack);
                                            }
                                            continue 'outer;
                                        }
                                    }
                                    _ => {}
                                }
                            }
                            break;
                        }
                        Packet::List(retval_inner)
                    }
                    c => Packet::Val(s.parse::<i32>().unwrap()),
                }
            })
        }
    }
}

fn read_input_file<P>(p: P) -> Result<Vec<(String, String)>>
where
    P: AsRef<Path>,
{
    let file = File::open(p)?;
    let reader = io::BufReader::new(file);

    Ok(reader
        .lines()
        .filter_map(|s| s.ok())
        .group_by(|s| !s.is_empty())
        .into_iter()
        // .filter(|(key, _)| *key)
        // .map(|(_, group)| group.collect_vec())
        .filter_map(|(key, group)| match key {
            true => {
                let group = group.collect_vec();
                let mut group = group.into_iter();
                Some((group.next()?, group.next()?))
            }
            false => None,
        })
        .collect_vec())
    // .for_each(|thing| println!("{:?}", thing));
}

fn parse(grouped_input: Vec<(String, String)>) -> Vec<(Packet, Packet)> {
    grouped_input
        .into_iter()
        .map(|(s1, s2)| {
            (
                Packet::from_str(&s1).unwrap(),
                Packet::from_str(&s2).unwrap(),
            )
        })
        .collect_vec()
}

fn main() -> Result<()> {
    env_logger::init();
    // too lazy to look up how to access args actually
    let input_arg = {
        let mut flup = std::env::args().into_iter();
        flup.next();
        flup.next()
    };
    let grouped_input =
        read_input_file(input_arg.ok_or(anyhow!("Something wrong reading command line args"))?)?;
    let parsed_input = parse(grouped_input);

    let result = parsed_input
        .par_iter()
        // .iter()
        .positions(|(first, second)| {
            let retval = first.partial_cmp(second);
            debug!("Comparing {:?} to {:?} yields {:?}", first, second, retval);
            info!("{:?}", retval);
            return retval == Some(Ordering::Less);
        })
        .map(|x| x + 1)
        // .collect_vec();
        .reduce(|| 0, |a, b| a + b);
    // .reduce(|a, b| a + b);

    println!("part 1 {:?}", result);

    let dividers = [
        Packet::List(vec![Packet::List(vec![Packet::Val(2)])]),
        Packet::List(vec![Packet::List(vec![Packet::Val(6)])]),
    ];

    let mut packets: Vec<Packet> = parsed_input
        .into_par_iter()
        .map(|(a, b)| vec![a, b])
        .flatten()
        .chain(dividers.clone())
        .collect();
    packets.par_sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let packets = packets;
    let packets = &packets;

    // let part2 = dividers
    //     .into_par_iter()
    //     .map(|divider| {
    //         packets
    //             .into_par_iter()
    //             .position_any(|inlist| *inlist == divider)
    //             .unwrap()
    //             + 1
    //     })
    //     .reduce(|| 1, |a, b| (a * b));

    let part2 = dividers
        .into_par_iter()
        .map(|divider| {
            for (i, other) in packets.into_iter().enumerate() {
                if *other == divider {
                    return i + 1;
                }
            }
            println!("aahahahahaah");
            0
        })
        .reduce(|| 1, |a, b| (a * b));

    println!("part 2 {}", part2);

    Ok(())
}
