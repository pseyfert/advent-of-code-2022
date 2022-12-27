#include <gtest/gtest.h>
#include "conv.h"

TEST(parse, example) {
  EXPECT_EQ(fromSNAFU("1=-0-2"), 1747ul);
  EXPECT_EQ(fromSNAFU("12111"), 906ul);
  EXPECT_EQ(fromSNAFU("2=0="), 198ul);
  EXPECT_EQ(fromSNAFU("21"), 11ul);
  EXPECT_EQ(fromSNAFU("2=01"), 201ul);
  EXPECT_EQ(fromSNAFU("111"), 31ul);
  EXPECT_EQ(fromSNAFU("20012"), 1257ul);
  EXPECT_EQ(fromSNAFU("112"), 32ul);
  EXPECT_EQ(fromSNAFU("1=-1="), 353ul);
  EXPECT_EQ(fromSNAFU("1-12"), 107ul);
  EXPECT_EQ(fromSNAFU("12"), 7ul);
  EXPECT_EQ(fromSNAFU("1="), 3ul);
  EXPECT_EQ(fromSNAFU("122"), 37ul);
}
