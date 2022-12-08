path is probably not well designed and should be split into iter and value, such that the iterator can be dereferenced and something be done with them.

also split internal logic better into header.

concept's are in the test commented out because they fail, that's not good.

path can't be used in subranges atm. that would be actually good

i didn't pay much attention with references and copying. probably there are way too many copies

for the actual analysis part, would be nice to use the const iterators because the directory tree doesn't change anymore.

and hey the root note is ugly af.
