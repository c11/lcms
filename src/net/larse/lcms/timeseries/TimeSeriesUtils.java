package net.larse.lcms.timeseries;

/**
 * Created by yang on 3/11/15.
 */


/**
 * Implements the EWMACD algorithm in the paper:
 * R.B. Cleveland, W.S.Cleveland, J.E. McRae, and I. Terpenning,
 * STL: A Seasonal-Trend Decomposition Procedure Based on Loess,
 * Statistics Research Report, AT&T Bell Laboratories.
 *
 * Based on the stl implementation from R package.
 *
 *
 * @auther Zhiqiang Yang, 3/11/2015
 *
 * TODO:
 *
 */

public class TimeSeriesUtils {


  /**
   *
   * @param y
   * @param n length of y
   * @param np
   * @param ns spans for s smoother
   * @param nt sapns for t smoother
   * @param nl spans for l smoother
   * @param isdeg local degree for s smoother
   * @param itdeg local degree for t smoother
   * @param ildeg local degree for l smoother
   * @param nsjump
   * @param ntjump
   * @param nljump
   * @param ni number of inner (robust) iterations
   * @param no number of outer (robust) iterations
   * @param rw
   * @param season
   * @param trend
   * @param work
   */
  public static void stl(double[] y, int n, int np,
                         int ns, int nt, int nl,
                         int isdeg, int itdeg, int ildeg,
                         int nsjump, int ntjump, int nljump,
                         int ni, int no,
                         double[] rw, double[] season,
                         double[] trend, double[] work) {


    // Original Fortran declaration
    // double precision y(n), rw(n), season(n), trend(n),
    // work(n+2*np,5)

    //TODO: should we implement work as 1d or 2d array

    // TODO: check how to initialize those return varaibles
    // In Fortran code, they are initialized by the caller,
    // in this implementation, should we change it?

    boolean userw = false;

    for (int i = 0; i < y.length; i++) {
      trend[i] = 0.0;
    }

    //the three spans must be at least three and odd
    int newns = Math.max(3, ns);
    int newnt = Math.max(3, nt);
    int newnl = Math.max(3, nl);

    if (newns % 2 == 0) {
      newns++;
    }

    if (newnt % 2 == 0) {
      newnt++;
    }

    if (newnl % 2 == 0) {
      newnl++;
    }

    int newnp = Math.max(2, np); //periodicity at least 2

    int k = 0;

    //outer loop -- robustness iterations
    while (k++ <= no) {
      //TODO: implement stlstp
      stlstp();

      //NOTE: assume using 1d array to represent 2d array for work
      for (int i = 0; i < y.length; i++) {
        work[i] = trend[i] + season[i];
      }

      //TODO: implement stlrwt
      stlrwt();

      userw = true;
    }

    //robustness weights when there were no robustness iterations
    if (no <= 0) {
      for (int i = 0; i < y.length; i++) {
        rw[i] = 1.0;
      }
    }
    return;
  }

  //TODO: parameter n should be removed as it is always the length of y
  public static void stlstp(double[] y, int n, int np, int ns, int nt, int nl,
                            int isdeg, int itdeg, int ildeg,
                            int nsjump, int ntjump, int nljump,
                            int ni, boolean userw,
                            double[] season, double[] trend, double[] work) {

    // From Fortran
    //integer n,np,ns,nt,nl,isdeg,itdeg,ildeg,nsjump,ntjump,nljump,ni
    //logical userw
    //double precision y(n),rw(n),season(n),trend(n),work(n+2*np,5)

    int baseWorkSize = y.length + 2 * np;

    for (int j = 0; j < ni; j++) {
      for (int i = 0; i < y.length; i++ ) {
        work[i] = y[i] - trend[i];
      }

      //TODO: implement stlss
      stlss();

      //TODO: implement stlfts
      stlfts();

      //TODO: implement stless
      stless();

      for (int i = 0; i < y.length; i++) {
        season[i] = work[baseWorkSize + np + i] - work[i];
        work[i] = y[i] - season[i];
      }

      stless();
    }
  }

  public static void stlrwt() {

  }

  public static void stlss(double[] y, int np, int ns, int isdeg, int nsjump,
                           boolean userw, double[] rw, double[] season,
                           double[] work1, double[] work2, double[] work3, double[] work4) {

    // From Fortran
    //integer n, np, ns, isdeg, nsjump
    //double precision y(n), rw(n), season(n+2*np),
    //&     work1(n), work2(n), work3(n), work4(n)
    //logical userw

    if (np < 1) {
      return;
    }

    for (int j = 0; j < np; j++) {
      int k = (y.length - j) / np + 1;

      for (int i = 0; i < k; i++) {
        work1[i] = y[i*np + j];
      }

      if (userw) {
        for (int i = 0; i < k; i++) {
          work3[i] = rw[i * np + j];
        }
      }

      stless();

      double xs = 0;
      int nright = Math.min(ns, k);

      boolean ok = stlest();

      if (!ok) {
        work2[0] = work2[1];
      }

      xs = k + 1;

      int nleft = Math.min(1, k-ns+1);

      ok = stlest();

      if (!ok) {
        //Fortran: if(.not. ok) work2(1) = work2(2)
        work2[k+1] = work2[1];
      }

      for (int m = 0; m < k+2; m++) {
        season[m * np + j] = work2[m];
      }
    }
  }



}
