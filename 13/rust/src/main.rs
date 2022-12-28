use env_logger;
use log::debug;

#[derive(Debug)]
enum Packet {
    Val(i32),
    List(Vec<Packet>),
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
                                    // TODO: logic error whether I push on ] or ,
                                    ']' => {
                                        if parenthesis_stack == 0 {
                                            let pack;
                                            (pack, to_be_packed) = to_be_packed.split_at(i);
                                            (_, to_be_packed) = to_be_packed.split_at(1);
                                            debug!("sending '{}' to recursion due to ]", pack);
                                            debug!("keeping '{}' on the stack", to_be_packed);
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
                                            debug!("sending '{}' to recursion due to ,", pack);
                                            debug!("keeping '{}' on the stack", to_be_packed);
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
                    c => Packet::Val(c.parse::<i32>().unwrap()),
                }
            })
        }
    }
}

fn main() {
    env_logger::init();
    for (argi, argument) in std::env::args().enumerate() {
        if 0 == argi {
            continue;
        }
        println!("{} → {:?}", &argument, Packet::from_str(&argument).unwrap());
    }
}
