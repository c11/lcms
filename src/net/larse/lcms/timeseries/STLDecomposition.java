package net.larse.lcms.timeseries;

/**
 * Created by yang on 3/12/15.
 *
 * Implement Seasonal Decomposition of Time Series by Loess
 *
 * R. B. Cleveland, W. S. Cleveland, J.E. McRae, and I. Terpenning (1990) STL:
 * A Seasonal-Trend Decomposition Procedure Based on Loess. Journal of Official Statistics, 6, 3–73.
 */
public class STLDecomposition {
  private double[] x;
  private double[] y;
  private int period;

  public STLDecomposition(double[] x, double[] y, int period) {
    this.x = x;
    this.y = y;
    this.period = period;
  }

  public double[] getTrend() {

    return null;
  }

  public double[] getSeasonal() {

    return null;
  }

  public double[] getRandom() {

    return null;
  }


  private double[] smooth(){
    return null;
  }

}
