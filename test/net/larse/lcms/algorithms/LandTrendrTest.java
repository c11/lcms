package net.larse.lcms.algorithms;

import it.unimi.dsi.fastutil.doubles.DoubleArrayList;
import org.junit.After;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;

import static org.junit.Assert.assertTrue;

public class LandTrendrTest {
  double[] x;
  double[] y;
  double[] vertices;
  double[] expected;

  LandTrendr.LandTrendrSolver lts;

  @Before
  public void setUp() throws Exception {
    lts = new LandTrendr.LandTrendrSolver();
  }

  @After
  public void tearDown() throws Exception {
    x = y = vertices = expected = null;
    lts = null;
  }

  @Test
  public void testGetResult2() throws Exception {
    x = new double[] {1984.0, 1985.0, 1986.0, 1987., 1988., 1989., 1990.,
        1991., 1992., 1993., 1994., 1995., 1996., 1997., 1998., 1999., 2000.,
        2001., 2002., 2003., 2004., 2005., 2006., 2007., 2008., 2009., 2010.,
        2011., 2012.};
    vertices = new double[] {1984, 1988, 1992, 2012};
    expected = new double[] {855, 626, 397, 168, -61, 28, 117, 206, 295, 316, 338, 358, 379,
        401, 422, 443, 464, 485, 507, 528, 549, 570, 591, 613, 634, 655, 676, 697, 719};
    y = new double[] {810, 686, 0, 0, -77, -27, 0, 184, 326, 361, 0, 0, 464, 507,
        540, 501, 0, 0, 530, 0, 561, 637, 633, 668, 562, 591, 644, 729, 611};

    double[] ftv = lts.getFTVResult(new DoubleArrayList(vertices), new DoubleArrayList(x), new DoubleArrayList(y));

    //due to rounding error between Java and IDL, 
    //allow the results to be different by 3 units in these testing,
    //that is 0.0003 difference in reflectance
    assertTrue(compareArray(ftv, expected, 3));
  }

  @Test
  public void testGetResult1() throws Exception {
    x = new double[] {1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998,
        1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013};

    y = new double[] {-215, -217,-173,-225,-242,-286,-429,-337,-376,-589,-544,-492,-516,-632,-615,-533,
        -619,-682,-614,-757,-726,-812,-586,-777,-720,-335,-434,-730,-706};

    vertices = new double[] {1985, 2009, 2010, 2013};

    expected = new double[] {-207.8, -232.9, -258.1, -283.2, -308.3,
        -333.4, -358.5, -383.7, -408.8, -433.9,
        -459.0, -484.2, -509.3, -534.4, -559.5,
        -584.7, -609.8, -634.9, -660.0, -685.1,
        -710.3, -735.4, -760.5, -785.6, -810.8,
        -335.0, -478.0, -621.0, -764.0};

    double[] ftv = lts.getFTVResult(new DoubleArrayList(vertices), new DoubleArrayList(x), new DoubleArrayList(y));

    //due to rounding error between Java and IDL, 
    //allow the results to be different by 3 units in these testing,
    //that is 0.0003 difference in reflectance
    assertTrue(compareArray(ftv, expected, 3));
  }

  private boolean compareArray(double[] result, double[] expected, double epsilon) {
    boolean pass = true;

    if (result.length != expected.length) {
      pass = false;
    }
    else {
      for (int i = 0; i < result.length; i++) {
        if (Math.abs(result[i] - expected[i]) > epsilon) {
          pass = false;
          break;
        }
      }
    }
    return pass;
  }
}