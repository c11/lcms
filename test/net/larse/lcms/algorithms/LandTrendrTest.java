package net.larse.lcms.algorithms;

import it.unimi.dsi.fastutil.doubles.DoubleArrayList;
import org.junit.After;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;

import java.util.Arrays;

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
    y = new double[] {810, 686, Double.NaN, Double.NaN, -77, -27, Double.NaN, 184, 326, 361, Double.NaN, Double.NaN, 464, 507,
        540, 501, Double.NaN, Double.NaN, 530, Double.NaN, 561, 637, 633, 668, 562, 591, 644, 729, 611};

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


  @Test
  public void testGetResultStartYearBeforeFirstVertex() throws Exception {
    x = new double[] {1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998,
            1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013};

    y = new double[] {-223,-215, -217,-173,-225,-242,-286,-429,-337,-376,-589,-544,-492,-516,-632,-615,-533,
            -619,-682,-614,-757,-726,-812,-586,-777,-720,-335,-434,-730,-706};

    vertices = new double[] {1985, 2009, 2010, 2013};

    expected = new double[] {-207.8, -207.8, -232.9, -258.1, -283.2, -308.3,
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

  @Test
  public void testGetResultStartYearAfterFirstVertex() throws Exception {
    x = new double[] {1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998,
            1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013};

    y = new double[] {-173,-225,-242,-286,-429,-337,-376,-589,-544,-492,-516,-632,-615,-533,
            -619,-682,-614,-757,-726,-812,-586,-777,-720,-335,-434,-730,-706};

    vertices = new double[] {1985, 2009, 2010, 2013};

    expected = new double[] {-258.1, -283.2, -308.3,
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

  @Test
  public void testGetResultEndYearAfterLastVertex() throws Exception {
    x = new double[] {1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998,
            1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014};

    y = new double[] {-215, -217,-173,-225,-242,-286,-429,-337,-376,-589,-544,-492,-516,-632,-615,-533,
            -619,-682,-614,-757,-726,-812,-586,-777,-720,-335,-434,-730,-706, -715};

    vertices = new double[] {1985, 2009, 2010, 2013};

    expected = new double[] {-207.8, -232.9, -258.1, -283.2, -308.3,
            -333.4, -358.5, -383.7, -408.8, -433.9,
            -459.0, -484.2, -509.3, -534.4, -559.5,
            -584.7, -609.8, -634.9, -660.0, -685.1,
            -710.3, -735.4, -760.5, -785.6, -810.8,
            -335.0, -478.0, -621.0, -764.0, -764.0};

    double[] ftv = lts.getFTVResult(new DoubleArrayList(vertices), new DoubleArrayList(x), new DoubleArrayList(y));

    //due to rounding error between Java and IDL,
    //allow the results to be different by 3 units in these testing,
    //that is 0.0003 difference in reflectance
    assertTrue(compareArray(ftv, expected, 3));
  }

  @Test
  public void testGetResultEndYearBeforeLastVertex() throws Exception {
    x = new double[] {1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998,
            1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011};

    y = new double[] {-215, -217,-173,-225,-242,-286,-429,-337,-376,-589,-544,-492,-516,-632,-615,-533,
            -619,-682,-614,-757,-726,-812,-586,-777,-720,-335,-434};

    vertices = new double[] {1985, 2009, 2010, 2013};

    expected = new double[] {-207.8, -232.9, -258.1, -283.2, -308.3,
            -333.4, -358.5, -383.7, -408.8, -433.9,
            -459.0, -484.2, -509.3, -534.4, -559.5,
            -584.7, -609.8, -634.9, -660.0, -685.1,
            -710.3, -735.4, -760.5, -785.6, -810.8,
            -335.0, -434.0};

    double[] ftv = lts.getFTVResult(new DoubleArrayList(vertices), new DoubleArrayList(x), new DoubleArrayList(y));

    //due to rounding error between Java and IDL,
    //allow the results to be different by 3 units in these testing,
    //that is 0.0003 difference in reflectance
    assertTrue(compareArray(ftv, expected, 3));
  }

  @Test
  public void testGetResultEndYearBeforeLastVertex2() throws Exception {
    x = new double[] {1984.0, 1985.0, 1986.0, 1987., 1988., 1989., 1990.,
            1991., 1992., 1993., 1994., 1995., 1996., 1997., 1998., 1999., 2000.,
            2001., 2002., 2003., 2004., 2005., 2006., 2007., 2008.};
    vertices = new double[] {1984, 1988, 1992, 2012};
    expected = new double[] {855, 626, 397, 168, -61, 28, 117, 206, 295.5, 319.576,
            343.652, 367.728, 391.804, 415.88, 439.956, 464.032, 488.108, 512.184,
            536.26, 560.336, 584.412, 608.488, 632.564, 656.64, 680.716};
    y = new double[] {810, 686, Double.NaN, Double.NaN, -77, -27, Double.NaN, 184, 326, 361, Double.NaN, Double.NaN, 464, 507,
            540, 501, Double.NaN, Double.NaN, 530, Double.NaN, 561, 637, 633, 668, 562};

    double[] ftv = lts.getFTVResult(new DoubleArrayList(vertices), new DoubleArrayList(x), new DoubleArrayList(y));

    //due to rounding error between Java and IDL,
    //allow the results to be different by 3 units in these testing,
    //that is 0.0003 difference in reflectance
    assertTrue(compareArray(ftv, expected, 3));
  }

  @Test
  /** missing first year **/
  public void testGetResultMissingFirstYear() throws Exception {
    x = new double[] {1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998,
            1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013};

    y = new double[] {Double.NaN, -215, -217,-173,-225,-242,-286,-429,-337,-376,-589,-544,-492,-516,-632,-615,-533,
            -619,-682,-614,-757,-726,-812,-586,-777,-720,-335,-434,-730,-706};

    vertices = new double[] {1985, 2009, 2010, 2013};

    expected = new double[] {-207.8, -207.8, -232.9, -258.1, -283.2, -308.3,
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

  @Test
  /** missing last year **/
  public void testGetResultMissingLastYear() throws Exception {
    x = new double[] {1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998,
            1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014};

    y = new double[] {-215, -217,-173,-225,-242,-286,-429,-337,-376,-589,-544,-492,-516,-632,-615,-533,
            -619,-682,-614,-757,-726,-812,-586,-777,-720,-335,-434,-730,-706, Double.NaN};

    vertices = new double[] {1985, 2009, 2010, 2013};

    expected = new double[] {-207.8, -232.9, -258.1, -283.2, -308.3,
            -333.4, -358.5, -383.7, -408.8, -433.9,
            -459.0, -484.2, -509.3, -534.4, -559.5,
            -584.7, -609.8, -634.9, -660.0, -685.1,
            -710.3, -735.4, -760.5, -785.6, -810.8,
            -335.0, -478.0, -621.0, -764.0, -764.0};

    double[] ftv = lts.getFTVResult(new DoubleArrayList(vertices), new DoubleArrayList(x), new DoubleArrayList(y));

    //due to rounding error between Java and IDL,
    //allow the results to be different by 3 units in these testing,
    //that is 0.0003 difference in reflectance
    assertTrue(compareArray(ftv, expected, 3));
  }

  @Test
  /** missing first three year **/
  public void testGetResultMissingFirstThreeYears() throws Exception {
    x = new double[] {1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998,
            1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013};

    y = new double[] {Double.NaN, Double.NaN, Double.NaN, -215, -217,-173,-225,-242,-286,-429,-337,-376,-589,-544,-492,-516,-632,-615,-533,
            -619,-682,-614,-757,-726,-812,-586,-777,-720,-335,-434,-730,-706};

    vertices = new double[] {1985, 2009, 2010, 2013};

    expected = new double[] {-207.8, -207.8, -207.8, -207.8, -232.9, -258.1, -283.2, -308.3,
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

  @Test
  /** not enough observation **/
  public void testGetResultNotEnoughObservations() throws Exception {
    x = new double[] {1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998,
            1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013};

    y = new double[] {Double.NaN, Double.NaN, Double.NaN, -215, Double.NaN,Double.NaN,Double.NaN,Double.NaN,
            -286,Double.NaN,Double.NaN,Double.NaN,Double.NaN,Double.NaN,Double.NaN,Double.NaN,Double.NaN,-615,-533,
            Double.NaN,Double.NaN,Double.NaN,Double.NaN,Double.NaN,Double.NaN,Double.NaN,Double.NaN,Double.NaN,
            Double.NaN,Double.NaN,Double.NaN,-706};

    vertices = new double[] {1985, 2009, 2010, 2013};

    expected = new double[x.length] ;
    Arrays.fill(expected, Double.NaN);

    double[] ftv = lts.getFTVResult(new DoubleArrayList(vertices), new DoubleArrayList(x), new DoubleArrayList(y));

    //due to rounding error between Java and IDL,
    //allow the results to be different by 3 units in these testing,
    //that is 0.0003 difference in reflectance
    assertTrue(compareArray(ftv, expected, 3));
  }

  @Test
  /** not enough observation **/
  public void testGetResultNotOverlapWithVertex() throws Exception {
    x = new double[] {1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990};

    y = new double[] {-215, -217,-173,-225,-242,-286,-429,-337,-376};

    vertices = new double[] {1990, 2009, 2010, 2013};

    expected = new double[x.length] ;
    Arrays.fill(expected, Double.NaN);

    double[] ftv = lts.getFTVResult(new DoubleArrayList(vertices), new DoubleArrayList(x), new DoubleArrayList(y));

    //due to rounding error between Java and IDL,
    //allow the results to be different by 3 units in these testing,
    //that is 0.0003 difference in reflectance
    assertTrue(compareArray(ftv, expected, 3));
  }

  @Test
  public void testGetResultOnlyTwoVertices() throws Exception {
    x = new double[] {1992., 1993., 1994., 1995., 1996., 1997., 1998., 1999., 2000.,
            2001., 2002., 2003., 2004., 2005., 2006., 2007., 2008., 2009., 2010.,
            2011., 2012.};
    vertices = new double[] {1984, 1988, 1992, 2012};
    expected = new double[] {391.05, 405.786, 420.522, 435.258, 449.994, 464.73,
            479.466, 494.202, 508.938, 523.674, 538.41, 553.146, 567.882, 582.618,
            597.354, 612.09, 626.826, 641.562, 656.298, 671.034, 685.77};
    y = new double[] {326, 361, Double.NaN, Double.NaN, 464, 507,
            540, 501, Double.NaN, Double.NaN, 530, Double.NaN, 561, 637, 633, 668, 562, 591, 644, 729, 611};

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