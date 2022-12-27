#include <gtest/gtest.h>
#include "conv.h"

TEST(parse, example) {
  // clang-format off
  EXPECT_EQ(fromSNAFU("1=-0-2"), 1747ul);
  EXPECT_EQ(fromSNAFU("12111" ),  906ul);
  EXPECT_EQ(fromSNAFU("2=0="  ),  198ul);
  EXPECT_EQ(fromSNAFU("21"    ),   11ul);
  EXPECT_EQ(fromSNAFU("2=01"  ),  201ul);
  EXPECT_EQ(fromSNAFU("111"   ),   31ul);
  EXPECT_EQ(fromSNAFU("20012" ), 1257ul);
  EXPECT_EQ(fromSNAFU("112"   ),   32ul);
  EXPECT_EQ(fromSNAFU("1=-1=" ),  353ul);
  EXPECT_EQ(fromSNAFU("1-12"  ),  107ul);
  EXPECT_EQ(fromSNAFU("12"    ),    7ul);
  EXPECT_EQ(fromSNAFU("1="    ),    3ul);
  EXPECT_EQ(fromSNAFU("122"   ),   37ul);
  // clang-format on
}

TEST(write, example) {
  // clang-format off
  EXPECT_EQ("1=-0-2", toSNAFU(1747ul));
  EXPECT_EQ("12111" , toSNAFU( 906ul));
  EXPECT_EQ("2=0="  , toSNAFU( 198ul));
  EXPECT_EQ("21"    , toSNAFU(  11ul));
  EXPECT_EQ("2=01"  , toSNAFU( 201ul));
  EXPECT_EQ("111"   , toSNAFU(  31ul));
  EXPECT_EQ("20012" , toSNAFU(1257ul));
  EXPECT_EQ("112"   , toSNAFU(  32ul));
  EXPECT_EQ("1=-1=" , toSNAFU( 353ul));
  EXPECT_EQ("1-12"  , toSNAFU( 107ul));
  EXPECT_EQ("12"    , toSNAFU(   7ul));
  EXPECT_EQ("1="    , toSNAFU(   3ul));
  EXPECT_EQ("122"   , toSNAFU(  37ul));
  // clang-format on
}
