package net.larse.lcms.algorithms;

import java.util.Arrays;
import java.util.List;
import java.util.TreeSet;
import com.google.common.primitives.Doubles;
import net.larse.lcms.helper.AlgorithmBase;

/**
 * Implements the Vegetation Change Tracker (VCT) that was proposed in the
 * paper: Huang C, SN Goward JG Masek, N Thomas, Z Zhu, JE Vogelmann (2010). An
 * automated approach for reconstructing recent forest disturbance history using
 * dense Landsat time series stacks. Remote Sensing of Environment, 114(1),
 * 183-198
 * 
 * @author Matt Gregory
 */
public final class VCT {
  // <editor-fold defaultstate="collapsed" desc=" CONSTANTS ">

  // Quality assurance (per pixel-year) constants
  private static final byte QA_BAD = 0;
  private static final byte QA_GOOD = 1;

  // Band constants - note these are unorthodox because we are using a 
  // subset of the TM bands for change detection
  private static final byte B3 = 0;
  private static final byte B4 = 1;
  private static final byte B5 = 2;
  private static final byte B7 = 3;
  private static final byte BT = 4;
  private static final byte NDVI = 5;
  private static final byte DNBR = 6;
  private static final byte R45 = 7;
  private static final byte COMP = 8;
  private static final byte N_BANDS = 9;

  // Mask constants
  private static final int WATER = 1;
  private static final int SHADOW = 2;
  private static final int SHADOW_EDGE = 3;
  private static final int CLOUD_EDGE = 4;
  private static final int CLOUD = 5;
  private static final int SNOW = 7;
  private static final int MASK_BAD_VALUE = 254;
  private static final int MASK_MISSING_VALUE = 255;

  // Disturbance land cover types
  private static final int PERM_NON_FOREST = 1;
  private static final int PERM_FOREST = 2;
  private static final int PART_FOREST = 3;
  private static final int PERM_WATER = 4;
  private static final int INTER_DISTURB_FOREST = 5;
  private static final int JUST_DISTURBED = 6;
  private static final int TRANSITION_PERIOD = 7;

  // Regrowth land cover types
  private static final int REGROWTH_NOT_OCCURRED = 1;
  private static final int REGROWTH_OCCURRED = 2;
  private static final int REGROWTH_TO_FOREST = 3;

  // Forest threshold maximum
  private static final double FOR_THR_MAX = 3.0;

  // Constants that signify consecutive and non-consecutive high and low UD
  private static final byte CHUD = 1;
  private static final byte CLUD = 2;
  private static final byte NCHUD = 3;
  private static final byte NCLUD = 4;
  // </editor-fold>
  
  // <editor-fold defaultstate="collapsed" desc=" VARIABLES ">

  // Arguments to class and class level variables
  private final Args args;
  private final double maxUd;
  private final double minNdvi;

  // Input variables
  private double[][] ud;         // Z-scores for all bands and indices
  private int[] mask;            // Mask (categorical) values for all years
  private int[] yearTable;       // Array to hold all years for this series
  private int numYears;          // Number of years (len of most arrays)

  // Characterization of input variables
  private byte[] qFlag;          // QA flag for each year in the series
  private int pctGoodObs;        // Percent of good QA flags
  private double uRange;         // Range of ud composite values
  private double vRange;         // Range of ud NDVI values
  private double fiRough;        // Roughness of forest index from composite
  private double fiRange;        // Range of forest index (= uRange)
  private double globalUdR2;     // R2 of entire series in ud B5
  private double globalUdSlp;    // Slope of entire series in ud B5

  // Attributes associated with maximum forest length segment - used
  // in thresholding
  private int maxConsFor;        // Maximum consecutive forest length
  private double[] meanForUdBx;  // Mean of index values across maxConsFor
  private double[] sdForUdBx;    // SD of index values across maxConsFor

  // Chararacterization of segments based on thresholded values
  private byte[] cstSeg;         // Segment characterization - (N)CHUD or (N)CLUD
  private byte[] cstSegSmooth;   // Smoothed characterization
  private int hudSeg;            // Number of high ud segments
  private int ludSeg;            // Number of low ud segments
  private int sharpTurns;        // Number of sharp turns between segments

  // Disturbance variables
  private int numDist;           // Number of disturbances detected
  private int numMajorDist;      // Number of major disturbances detected
  private int longestDistLength; // Length of the longest disturbance
  private double longestDistR2;  // R2 of the longest disturbance
  private double longestDistRough; // Roughness of the longest disturbance
  private byte[] distFlag;       // Type of each disturbance
  private int[] distYear;        // Year of each disturbance
  private byte[] distLength;     // Length of each disturbance
  private double[] distR2;       // R2 of each disturbance
  private double[] distMagn;     // Magnitude (in ud composite) of each dist.
  private double[] distMagnB4;   // Magnitude (in ud B4) of each dist.
  private double[] distMagnVi;   // Magnitude (in ud NDVI) of each dist.
  private double[] distMagnBr;   // Magnitude (in ud DNBR) of each dist.

  // Regrowth variables
  private double[] regrR2;       // R2 of each regrowth
  private double[] regrSlope;    // Slope of each regrowth
  private double[] regrRough;    // Roughness of each regrowth
  private byte[] regrFlag;       // Type of each regrowth

  // Land cover/change types
  private int lcType;            // Final land cover/change type
  private int agIndicator;       // Agriculture indicator - set but not used
  // </editor-fold>      
  
  static class Args extends AlgorithmBase.ArgsBase {
    @Doc(help = "Maximum UD composite value for forest")
    @Optional
    double maxUd = 4.0;
    
    @Doc(help = "Mininum NDVI value for forest")
    @Optional
    double minNdvi = 0.45;
  }
    
  public VCT() {
    this.args = new Args();
    this.maxUd = args.maxUd;
    this.minNdvi = args.minNdvi;
  }
  
  public VCT(Args args) {
    this.args = args;
    this.maxUd = args.maxUd;
    this.minNdvi = args.minNdvi;
  }
  
  /**
   * This is the main function for determining change analysis within VCT.
   * First, time series information is read from a number of different forest
   * indexes (known as forest z-scores in the paper, ud scores throughout the
   * code) and a mask image denoting land cover type and/or image artifacts
   * (e.g. cloud, shadow). The algorithm consists of two main functions and a
   * cleanup function:
   *
   * 1) interpolationAndIndices - The ud series for all indices is first filled
   * in for all cloud/shadow pixels from neighboring years.
   *
   * 2) analyzeUDist - The filled series are analyzed to find anomalous
   * deviations from neighboring years and characterized into disturbance and
   * regrowth segments
   *
   * 3) setDisturbanceVariables - Various cleanup and clamping of variable
   * values as well as determining disturbance duration.
   *
   * Note that much of the code relies of comparing series values with known or
   * derived thresholds of "forestness" and anomalies are detected when these
   * threshold bounds are exceeded.
   *
   * @param ud - List of all UD values across all pertinent bands and indices
   * (B3, B4, B5, B7, thermal, NDVI, DNBR, R45). There is one List per index
   * which contains values for all years.
   *
   * @param mask - List of mask values for this pixel across all years
   *
   * @param years - List of years corresponding to the indices in ud and mask
   *
   * @return - Unspecified for now
   */
  public List<List<Double>> getResult(List<List<Double>> ud,
          List<Integer> mask, List<Integer> years) {
    initializePixel(ud, mask, years);
    interpolationAndIndices();
    analyzeUDist();
    setDisturbanceVariables();

    // TODO: Not sure what we want to return at this point.  Leave null
    // for now.
    return null;
  }

  /**
   * Initialize the pixel's values from the passed arguments and calculate the
   * composite UD score as a function of bands 3, 5, and 7
   * 
   * @param ud - List of all UD values across all pertinent bands and indices
   * (B3, B4, B5, B7, thermal, NDVI, DNBR, R45). There is one List per index
   * which contains values for all years.
   * 
   * @param mask - List of mask values for this pixel across all years
   * 
   * @param years - List of years corresponding to the indices in ud and mask
   */
  private void initializePixel(List<List<Double>> ud, List<Integer> mask,
          List<Integer> years) {

    int i, j;

    this.numYears = ud.get(0).size();
    this.ud = new double[N_BANDS][this.numYears];
    this.mask = new int[this.numYears];
    this.yearTable = new int[this.numYears];

    // Initialize the UD values from parameters, scaling and translating
    // per variable
    // 
    // TODO: need better way to do this -- too fragile
    // This assumes that the input data is stored by bands and then years
    // with the following bands (or indexes): B3, B4, B5, B7, BT (thermal),
    // NDVI, NBR, R45.
    final double SCALE = 0.01;
    final double OFFSET = -100.5;
    double[] s = {SCALE, SCALE, SCALE, SCALE, SCALE, SCALE, SCALE, 1.0};
    double[] t = {0.0, 0.0, 0.0, 0.0, 0.0, OFFSET, OFFSET, 0.0};
    for (i = 0; i < this.numYears; i++) {
      for (j = 0; j < N_BANDS - 1; j++) {
        this.ud[j][i] = s[j] * (ud.get(j).get(i) + t[j]);
      }
      this.mask[i] = mask.get(i);
      this.yearTable[i] = years.get(i);
    }

    // Calculate the composite UD variable
    double tmp, sumSq;
    List<Byte> indexes = Arrays.asList(B3, B5, B7);
    for (i = 0; i < this.numYears; i++) {
      sumSq = 0.0;
      for (Byte index : indexes) {
        tmp = this.ud[index][i];
        tmp = tmp >= 0.0 ? tmp : tmp / 2.5;
        sumSq += tmp * tmp;
      }
      this.ud[COMP][i] = Math.sqrt(sumSq / indexes.size());
    }
  }

  /**
   * Classify the pixel's mask values into QA_GOOD and QA_BAD and fill in 
   * QA_BAD values based on nearest neighbors or linear interpolation of 
   * QA_GOOD values
   */
  private void interpolationAndIndices() {

    // Implements section 3.3.1 in the Cheng et al. (2010) paper
    int i, j, k, n, prev, next;
    int badCount = 0;

    // QA flag - whether or not a pixel is suitable for interpolation
    this.qFlag = new byte[this.numYears];

    // Create a set which represents invalid mask values for interpolation
    TreeSet INVALID = new TreeSet<>();
    INVALID.add(CLOUD);
    INVALID.add(CLOUD_EDGE);
    INVALID.add(SHADOW);
    INVALID.add(SHADOW_EDGE);
    INVALID.add(MASK_MISSING_VALUE);
    INVALID.add(SNOW);
    INVALID.add(MASK_BAD_VALUE);

    // Flag each year's pixel as good or bad based on the mask value
    for (i = 0; i < this.numYears; i++) {
      this.qFlag[i] = INVALID.contains(this.mask[i]) ? QA_BAD : QA_GOOD;
    }

    // Look for spikes and dips that may be unflagged cloud and shadow and
    // set the QA flag accordingly
    for (i = 1; i < this.numYears - 1; i++) {

      // Skip over pixels already labeled as QA_BAD
      if (this.qFlag[i] == QA_BAD) {
        continue;
      }

      // Pixels higher in UD than neighboring years - cloud
      if (isRelativeCloud(i)) {
        this.qFlag[i] = QA_BAD;
        continue;
      }

      // Pixels lower than UD than neighboring years - shadow
      if (isRelativeShadow(i)) {
        this.qFlag[i] = QA_BAD;
      }
    }

    // Now test the start/end conditions
    n = this.numYears;
    if (this.qFlag[0] == QA_GOOD && isBadEndpoint(0, 1)) {
      this.qFlag[0] = QA_BAD;
    }
    if (this.qFlag[n - 1] == QA_GOOD && isBadEndpoint(n - 1, n - 2)) {
      this.qFlag[n - 1] = QA_BAD;
    }

    // Calculate bad percentage
    for (i = 0; i < this.numYears; i++) {
      if (this.qFlag[i] == QA_BAD) {
        badCount += 1;
      }
    }
    this.pctGoodObs = (int) (100.0 - ((100.0 * badCount) / this.numYears));

    // Interpolate for bad observations indicated by qFlag
    i = 0;
    if (this.pctGoodObs > 50.0) {
      while (i < this.numYears) {
        // Skip good observations
        if (this.qFlag[i] == QA_GOOD) {
          i += 1;
        } // Fill or interpolate bad observations
        else {
              // Search forward/backward to find next valid observations in 
          // time series
          prev = i - 1;
          next = i + 1;
          while (prev >= 0 && this.qFlag[prev] == QA_BAD) {
            prev -= 1;
          }
          while (next < this.numYears && this.qFlag[next] == QA_BAD) {
            next += 1;
          }

          // No valid QA_GOOD pixels in the time series
          if (prev < 0 && next >= this.numYears) {
            break;
          } // No acceptable previous QA_GOOD - use next index to fill
          // all years from 0 to next
          else if (prev < 0) {
            for (j = 0; j < next; j++) {
              for (k = 0; k < N_BANDS; k++) {
                this.ud[k][j] = this.ud[k][next];
              }
            }
            i = next + 1;
          } // No acceptable next QA_GOOD - use prev index to fill
          // all years from prev + 1 to num_years 
          else if (next >= this.numYears) {
            for (j = prev + 1; j < this.numYears; j++) {
              for (k = 0; k < N_BANDS; k++) {
                this.ud[k][j] = this.ud[k][prev];
              }
            }
            i = next + 1;
          } // Found years acceptable for interpolation - fill between 
          // prev and next
          else {
            for (k = 0; k < N_BANDS; k++) {
              interpolateValues(this.ud[k], prev, next);
            }
            i = next + 1;
          }
        }
      }
    }

    // Get range values for the composite UD and NDVI
    this.uRange = Doubles.max(this.ud[COMP]) - Doubles.min(this.ud[COMP]);
    this.vRange = Doubles.max(this.ud[NDVI]) - Doubles.min(this.ud[NDVI]);

    this.fiRange = this.uRange;
    this.fiRough = fiRoughness(this.ud[COMP], 0, this.numYears);

    // TODO: This is a restriction to ensure that when scaled by 10.0, it 
    // doesn't exceed byte length - unneeded?
    if (this.fiRough > 25.4) {
      this.fiRough = 25.4;
    }
  }

  /**
   * Main function for determining disturbance and regrowth for this pixel's
   * trajectory.  Main steps include:
   * 
   *   1) Determine composite and NDVI thresholds
   *   2) Find the longest consecutive forest streak
   *   3) Find consecutive segments as either high or low UD values based on
   *      thresholds
   *   4) Smooth these segments
   *   5) Characterize segments as disturbance/regrowth classes
   *   6) Determine land cover/change type based on pattern of disturbances
   */
  private void analyzeUDist() {
    int i;

    // Initialize the disturbance and regrowth arrays
    this.distYear = new int[this.numYears];
    this.regrR2 = new double[this.numYears];
    this.regrSlope = new double[this.numYears];
    this.regrRough = new double[this.numYears];
    this.regrFlag = new byte[this.numYears];
    this.distLength = new byte[this.numYears];
    this.distR2 = new double[this.numYears];
    this.distMagn = new double[this.numYears];
    this.distMagnB4 = new double[this.numYears];
    this.distMagnVi = new double[this.numYears];
    this.distMagnBr = new double[this.numYears];

    // Assign some default information to this pixel
    this.distYear[0] = this.distYear[1] = this.numYears;
    this.lcType = PART_FOREST;

    // Get the top ndvi values and the bottom two ud composite values.  If
    // the spread between the low UD values is less than FOR_THR_MAX, set
    // them equal. From this point on, min2Ud is used instead of minUd to
    // reduce impact of an anomalously low ud value
    //  
    // TODO: As written, this is currently a bug in the original VCT
    // software - if the last number in the UD composite sequence is the
    // lowest number, min2Ud will stay as 9999.0 instead of the second
    // lowest number.  Keeping in here to for comparison purposes, but
    // should be changed.  The below (commented) section implements the
    // desired functionality
    // 
    // double maxVi = Doubles.max(this.ud[NDVI]);
    // double minUd = Doubles.min(this.ud[COMP]);
    // double min2Ud = Doubles.max(this.ud[COMP]);
    // for (i = 0; i < this.numYears; i++) {
    //   if (this.ud[COMP] > minUd && this.ud[COMP] < min2Ud) {
    //     min2Ud = this.ud[COMP];
    //   }
    // }
    double maxVi = Doubles.max(this.ud[NDVI]);
    double minUd = 9999.0;
    double min2Ud = 9999.0;
    double cmp;
    for (i = 0; i < this.numYears; i++) {
      cmp = this.ud[COMP][i];
      if (cmp < minUd) {
        minUd = cmp;
      }
      if (cmp > minUd && cmp < min2Ud) {
        min2Ud = cmp;
      }
    }
    if (min2Ud - minUd < FOR_THR_MAX) {
      min2Ud = minUd;
    }

    // Track the number of water and shadow pixels and identify if they
    // come in the first and last thirds of the time series
    int isWaterFront = 0;
    int isWaterTail = 0;
    int firstThird = this.numYears / 3;
    int lastThird = this.numYears - firstThird;
    double percentWater = 0.0;
    double percentShadow = 0.0;
    for (i = 0; i < this.numYears; i++) {
      if (this.mask[i] == WATER || this.mask[i] == SHADOW) {
        percentWater += 1.0;
        if (this.mask[i] == SHADOW) {
          percentShadow += 1.0;
        }
        isWaterFront += i < firstThird ? 1 : 0;
        isWaterTail += i >= lastThird ? 1 : 0;
      }
    }
    percentWater /= this.numYears;
    percentShadow /= this.numYears;

    // Get the maximum streak of years with forest.  The length of the
    // streak gets set in the function (this.maxConsFor) along with 
    // the mean and standard deviations of the longest streak in each
    // index.  The value returned is the starting year of the streak.
    int maxForStart;
    maxForStart = getMaxForLength(min2Ud, min2Ud + FOR_THR_MAX);

    // Set a threshold for determining change in the UD composite signal
    double changeThrUd;
    double changeHike = FOR_THR_MAX;
    double adjCoeff = this.meanForUdBx[COMP] / 5.0;
    adjCoeff = adjCoeff > 1.67 ? 1.67 : adjCoeff;
    if (adjCoeff > 1.0) {
      changeHike *= adjCoeff;
    }
    changeThrUd = this.meanForUdBx[COMP] + changeHike;

    // Identify consecutive high and low ud observations - this loop
    // characterizes each year in the trajectory into one of four
    // categories: consecutive low UD (CLUD), non-consecutive low UD
    // (NCLUD), consecutive high UD (CHUD), and non-consecutive high
    // UD (NCHUD). If a trajectory is above or below the dividing
    // threshold (change_thr_ud) for two or more years, it is called
    // CLUD or CHUD. If it only stays for one year, it is called NCLUD
    // or NCHUD.
    i = 0;
    int j, k;
    int numCstObs;
    this.cstSeg = new byte[this.numYears];
    this.ludSeg = this.hudSeg = this.sharpTurns = 0;
    while (i < this.numYears) {

      // Consecutive low ud - CLUD
      j = i;
      numCstObs = 0;
      while (j < this.numYears && this.ud[COMP][j] <= changeThrUd) {
        j += 1;
        numCstObs += 1;
      }
      if (numCstObs >= 2) {
        for (k = i; k < j; k++) {
          this.cstSeg[k] = CLUD;
        }
      } else {
        for (k = i; k < j; k++) {
          this.cstSeg[k] = NCLUD;
        }
      }
      if (numCstObs > 0) {
        this.ludSeg += 1;
        this.sharpTurns += 1;
      }

      // Consecutive high ud - CHUD
      i = j;
      numCstObs = 0;
      while (j < this.numYears && this.ud[COMP][j] > changeThrUd) {
        j += 1;
        numCstObs += 1;
      }
      if (numCstObs >= 2) {
        for (k = i; k < j; k++) {
          this.cstSeg[k] = CHUD;
        }
      } else {
        for (k = i; k < j; k++) {
          this.cstSeg[k] = NCHUD;
        }
      }
      if (numCstObs > 0) {
        this.hudSeg += 1;
        this.sharpTurns += 1;
      }
      i = j;
    }

    // Calculate fast greenup indicator of nonforest
    calculateAgIndicator();

    // Remove NCLUD and NCHUD labels based on adjacent labels - this 
    // effectively smooths the segment labels
    this.cstSegSmooth = new byte[this.numYears];
    for (i = 0; i < this.numYears; i++) {
      this.cstSegSmooth[i] = this.cstSeg[i];
    }
    smoothSegment(this.cstSegSmooth, this.numYears, NCLUD);
    smoothSegment(this.cstSegSmooth, this.numYears, NCHUD);

    // Create a time series instance to store segment information.  This
    // block uses the smoothed segment information to create the segments
    i = 0;
    TSSegment tsSeg = new TSSegment(this.numYears);
    int nSeg = 0;
    int segLength;
    while (i < this.numYears) {

      // Initialize this segment
      j = i;
      segLength = 1;

          // As long as the label for this year matches the following year's 
      // label, keep growing the segment
      while (j < this.numYears - 1
              && this.cstSegSmooth[j] == this.cstSegSmooth[j + 1]) {
        j++;
        segLength += 1;
      }

      // Store this segment
      tsSeg.segType[nSeg] = this.cstSegSmooth[i];
      tsSeg.startYear[nSeg] = (byte) i;
      tsSeg.endYear[nSeg] = (byte) (i + segLength - 1);
      tsSeg.segLength[nSeg] = (byte) segLength;

      // Increment for the next segment
      nSeg++;
      i = j + 1;
    }
    tsSeg.totSeg = nSeg;

    // Now detect changes
    // Note that ALL pixels go through this logic, although this
    // information is only used at the end where this.lcType == PART_FOREST
    int distNum;
    this.distFlag = new byte[this.numYears];
    for (nSeg = 0; nSeg < tsSeg.totSeg; nSeg++) {
      switch (tsSeg.segType[nSeg]) {

        // Consecutive high UD - signifies disturbance event
        case CHUD:
          this.numMajorDist += 1;

          // Characterize the disturbance and following recovery
          setDisturbance(tsSeg.startYear[nSeg], tsSeg.endYear[nSeg]);

          // More convenient to set regrowth type here
          // dist_num is decremented due to increment in set_disturbance
          distNum = this.numDist - 1;
          this.regrFlag[distNum] = REGROWTH_NOT_OCCURRED;

          // Not the last segment in this time series and is followed by
          // a forested segment
          if (nSeg < tsSeg.totSeg - 1) {
            if (tsSeg.segType[nSeg + 1] == CLUD
                    || tsSeg.segType[nSeg + 1] == NCLUD) {
              this.regrFlag[distNum] = REGROWTH_TO_FOREST;
            } else {
              // Handle exception
              // String msg = "Warning: CHUD not followed by CLUD or NCLUD";
              // throw new Exception(msg);
            }
          } // Last segment in this time series, but high R2 and
          // negative slope indicate that regrowth is occurring
          else if (this.regrR2[distNum] > 0.7
                  && this.regrSlope[distNum] < -0.2) {
            this.regrFlag[distNum] = REGROWTH_OCCURRED;
          }
          break;

        // Consecutive low UD - forested
        case CLUD:
          // Mark the pixel's distFlag for the years in the segment
          setPostDisturbForest(tsSeg.startYear[nSeg], tsSeg.endYear[nSeg]);

          // Search for low-level disturbance
          searchMinorDisturbances(tsSeg.startYear[nSeg], tsSeg.endYear[nSeg]);
          break;

        // Non-consecutive high UD
        case NCHUD:
          // End year of this sequence is last year in time series, mark
          // as disturbance
          if (tsSeg.endYear[nSeg] == numYears - 1) {
            setDisturbance(tsSeg.startYear[nSeg], tsSeg.endYear[nSeg]);
          } // Mark the pixel's distFlag for the years in the segment
          else {
            setPostDisturbForest(tsSeg.startYear[nSeg], tsSeg.endYear[nSeg]);
          }
          break;

        // Non-consecutive low UD
        case NCLUD:
          // Mark the pixel's distFlag for the years in the segment
          setPostDisturbForest(tsSeg.startYear[nSeg], tsSeg.endYear[nSeg]);
          break;

        default:
          this.lcType = PERM_NON_FOREST;
          break;
      }
    }

    // Run global linear fits across the time series, but leave off
    // portions of tails (three at both beginning and end) and
    // capture the best fit
    double[] coeff;
    for (i = 0; i < 3; i++) {
      coeff = calculateLinearChangeRate(i, this.numYears - 1);
      if (coeff[2] > this.globalUdR2) {
        this.globalUdR2 = coeff[2];
        this.globalUdSlp = coeff[0];
      }
    }

    for (i = this.numYears - 3; i < numYears; i++) {
      coeff = calculateLinearChangeRate(0, i);
      if (coeff[2] > this.globalUdR2) {
        this.globalUdR2 = coeff[2];
        this.globalUdSlp = coeff[0];
      }
    }

    // TODO: If this.lcType is driving the assignment and all other outputs
    // are predicated upon that designation, it seems like we can short
    // circuit the logic right after we calculate these variables
    
    // Final classification - set lcType based on calculated data
    // Water dominated pixel and present throughout the time series
    if (isWaterFront > 1 && isWaterTail > 1
            && percentWater > 0.4 && percentWater > percentShadow * 2.0) {
      this.lcType = PERM_WATER;
    }

    // Noisy time series (many sharp turns) - signifies ag or other
    // nonforest
    if (this.sharpTurns > (int) this.numYears * 0.33) {
      this.lcType = PERM_NON_FOREST;
    }

    // Minimum UD exceeds forest UD or maximum VI under forest NDVI
    // nonforest
    if (min2Ud > this.maxUd || maxVi < this.minNdvi) {
      this.lcType = PERM_NON_FOREST;
    }

    // Short duration segments where the maxForStart was not at the 
    // beginning or end of the time series - nonforest
    if (this.maxConsFor < 3 && maxForStart > 1
            && maxForStart < (this.numYears - 1 - this.maxConsFor)) {
      this.lcType = PERM_NON_FOREST;
    }

    // Seems to signify short duration segments as well
    if (this.maxConsFor < 1) {
      if (percentWater > 0.15) {
        this.lcType = PERM_WATER;
      } else {
        this.lcType = PERM_NON_FOREST;
      }
    }

    // Only one segment identfied - persistent forest
    if (this.numDist == 0) {
      this.lcType = PERM_FOREST;
    }
  }

  /**
   * Characterize longest disturbance segment and clamp disturbance variable
   * values
   */
  private void setDisturbanceVariables() {
    int i, dLength, longestDistIdx, first, last;

    if (this.uRange > 25.4) {
      this.uRange = 25.4;
    }

    for (i = 0; i < this.numYears; i++) {
      if (this.distMagn[i] > 25.0) {
        this.distMagn[i] = 25.0;
      }
    }

    if (this.lcType == PART_FOREST) {
      if (this.numDist < 1 || this.numDist >= this.numYears) {
            // Handle exception
        // String msg = "Error: Wrong number of disturbances";
        // throw new Exception(msg);
      }

      // Find the disturbance with longest duration
      dLength = 0;
      longestDistIdx = 0;
      for (i = 0; i < this.numDist; i++) {
        if (this.distLength[i] > dLength) {
          longestDistIdx = i;
          dLength = this.distLength[i];
        }
      }
      this.longestDistR2 = this.distR2[longestDistIdx];
      this.longestDistRough = this.regrRough[longestDistIdx];
      this.longestDistLength = dLength;

      // Clamp pixel values
      first = this.distYear[0];
      last = this.distYear[this.numDist - 1];

      this.distMagn[first]
              = Math.max(Math.min(this.distMagn[first], 25.0), 0.0);
      this.distMagn[last]
              = Math.max(Math.min(this.distMagn[last], 25.0), 0.0);

      this.distMagnVi[first]
              = Math.max(Math.min(this.distMagnVi[first], 1.0), -1.0);
      this.distMagnVi[last]
              = Math.max(Math.min(this.distMagnVi[last], 1.0), -1.0);

      this.distMagnBr[first]
              = Math.max(Math.min(this.distMagnBr[first], 1.0), -1.0);
      this.distMagnBr[last]
              = Math.max(Math.min(this.distMagnBr[last], 1.0), -1.0);
    }

    this.globalUdSlp = Math.max(Math.min(this.globalUdSlp, 1.0), -1.0);
  }

  /**
   * Given a time series and two endpoints, linearly interpolate all values
   * 
   * @param ts - Array of values to interpolate
   * @param left - Left endpoint to use
   * @param right - Right endpoint to use
   */
  private void interpolateValues(double[] ts, int left, int right) {
    int j;
    double denom, slope;

    denom = (double) (right - left);
    slope = (ts[right] - ts[left]) / denom;
    for (j = left + 1; j < right; j++) {
      ts[j] = ts[left] + slope * (j - left);
    }
  }

  /**
   * Calculate a measure of a time series' inter-annual variability
   *
   * @param ts - Array of time series values
   * @param left - Left endpoint to use
   * @param right - Right endpoint to use
   * @return - Roughness value
   */
  private double fiRoughness(double[] ts, int left, int right) {
    int numVals = right - left;

    if (numVals <= 3) {
      return -1.0;
    }

    // Find all differences between consecutive data pairs
    int i, tmpIdx;
    double[] tmpData = new double[numVals - 1];
    for (i = left; i < right - 1; i++) {
      tmpData[i] = ts[i + 1] - ts[i];
    }

        // Sort these differences and find the index 1/10 in from the left
    // constrained betweeen indexes 1 and 3 inclusive; 
    // return the absolute value of this pair's difference
    Arrays.sort(tmpData);
    tmpIdx = (int) (numVals * 0.1);
    if (tmpIdx < 1) {
      tmpIdx = 1;
    }
    if (tmpIdx > 3) {
      tmpIdx = 3;
    }
    return Math.abs(tmpData[tmpIdx]);
  }

  /**
   * Determine if a year's pixel value is relatively cloudy based on its
   * neighbors values
   * 
   * @param i - index to check
   * @return - cloudiness flag
   */
  private boolean isRelativeCloud(int i) {
    // Attempt at separating out complex logic into little-seen methods
    // However, this doesn't allow the code to short-circuit
    boolean cond = this.ud[COMP][i] > this.ud[COMP][i - 1] + 3.5;
    cond |= this.ud[COMP][i] > this.ud[COMP][i + 1] + 3.5;
    cond &= this.ud[COMP][i] > this.ud[COMP][i - 1] + 2.5;
    cond &= this.ud[COMP][i] > this.ud[COMP][i + 1] + 2.5;
    cond &= this.ud[BT][i] < this.ud[BT][i - 1] - 1.0;
    cond &= this.ud[BT][i] < this.ud[BT][i + 1] - 1.0;
    cond &= this.ud[BT][i] < 0.5;
    return cond;
  }

  /**
   * Determine if a year's pixel value is relatively shadow based on its
   * neighbors values
   * 
   * @param i - index to check
   * @return - shadow flag
   */
  private boolean isRelativeShadow(int i) {
    boolean cond = this.ud[COMP][i] > this.ud[COMP][i - 1] - 3.5;
    cond |= this.ud[COMP][i] > this.ud[COMP][i + 1] - 3.5;
    cond &= this.ud[COMP][i] < this.ud[COMP][i - 1] - 2.5;
    cond &= this.ud[COMP][i] < this.ud[COMP][i + 1] - 2.5;
    cond &= this.ud[B4][i] < 1.0;
    cond &= this.ud[B5][i] < 1.0;
    cond &= this.ud[B7][i] < 1.0;
    return cond;
  }

  /**
   * Determine if an endpoint is cloud/shadow based on neighbor's value
   * 
   * @param i - index to check
   * @param j - neighbor's index to check against
   * @return - bad flag
   */
  private boolean isBadEndpoint(int i, int j) {
    // Likely cloud
    boolean cond1 = this.ud[COMP][i] > this.ud[COMP][j] + 3.5;
    cond1 &= this.ud[BT][i] < this.ud[BT][j] - 1.5;
    cond1 &= this.ud[BT][i] < 0.5;

    // Likely shadow
    boolean cond2 = this.ud[COMP][i] < this.ud[COMP][j] - 3.5;
    cond2 &= this.ud[B5][i] < 1.0;
    cond2 &= this.ud[B7][i] < 1.0;
    cond2 &= this.ud[B4][0] < 1.0;

    return (cond1 || cond2);
  }

  /**
   * Get the maximum length of a pixel's forestness.  This is based on a 
   * consistent ud composite value within the given range
   * 
   * @param minForUd - Minimum ud composite value for forest
   * @param maxForUd - Maximum ud composite value for forest
   * @return - the number of years this pixel is in a forest condition
   */
  private int getMaxForLength(double minForUd, double maxForUd) {
    // Find the longest streak (maxLength) over the time series when ud is
    // persistently within the ud range.  For this streak, also track the
    // number WATER observations
    int i = 0;
    int j;
    int iStart = 0;
    int iEnd = 0;
    int waterCount = 0;
    int maxLength = 0;
    int yearCount, tmpYear, tmpWaterCount;

    while (i < this.numYears) {

      // Reset counts
      yearCount = 0;
      tmpYear = i;
      tmpWaterCount = 0;

      // Start tracking the streak
      while (tmpYear < this.numYears
              && this.ud[COMP][tmpYear] <= maxForUd
              && this.ud[COMP][tmpYear] >= minForUd) {
        if (this.mask[tmpYear] == WATER) {
          tmpWaterCount++;
        }
        tmpYear++;
        yearCount++;
      }

      // Check to see if the streak has been exceeded
      if (maxLength < yearCount) {
        maxLength = yearCount;
        waterCount = tmpWaterCount;
        iStart = i;
        iEnd = tmpYear;
      }

      // Move the year pointer ahead
      if (yearCount > 0) {
        i += yearCount;
      } else {
        i++;
      }
    }

    // Initialize the container to hold means and standard deviations of
    // the maximum forest streak
    this.meanForUdBx = new double[N_BANDS];
    this.sdForUdBx = new double[N_BANDS];
    for (i = 0; i < N_BANDS; i++) {
      this.meanForUdBx[i] = 25.4;
      this.sdForUdBx[i] = 25.4;
    }

    // If there is a forest streak, calculate statistics on this streak
    if (maxLength > 0) {
      for (j = 0; j < N_BANDS; j++) {
        this.meanForUdBx[j] = getSliceMean(this.ud[j], iStart, iEnd);
      }

      if (maxLength > 1) {
        for (j = 0; j < N_BANDS; j++) {
          this.sdForUdBx[j] = getSliceStd(this.ud[j], iStart, iEnd);
        }
      } else {
            // Calculate standard deviations using a high SD value if
        // maxLength is too short
        for (j = 0; j < N_BANDS; j++) {
          this.sdForUdBx[j] = this.meanForUdBx[j] / 3.0;
        }
      }
    }
    this.maxConsFor = maxLength - waterCount;
    return (iStart);
  }

  /**
   * Determine whether or not a pixel is indicative of agriculture - high
   * yearly variability in signal
   */
  private void calculateAgIndicator() {
    int i, udAllDrop, udB5Drop, udB4Hike, udDropNum;
    int udDip, udHike;
    double suddenChange = 3.5;

    this.agIndicator = 0;

    for (i = 1; i < this.numYears - 1; i++) {
      // Consective low UD (CLUD) vertices
      // If there is a sudden change in UD values in either B4, B5 or the
      // composite, calculate the number of drops (defined as the
      // magnitude of the change divided by the change threshold) in each
      // of B4, B5 and COMP.  Take the maximum of these drops, square
      // this value, and add to this.agIndicator
      boolean cond = this.ud[COMP][i - 1] - this.ud[COMP][i] > suddenChange;
      cond |= this.ud[B5][i - 1] - this.ud[B5][i] > suddenChange;
      cond |= this.ud[B4][i - 1] - this.ud[B4][i] > suddenChange;
      if (this.cstSeg[i] == CLUD && cond) {
        udAllDrop
                = (int) ((this.ud[COMP][i - 1] - this.ud[COMP][i]) / suddenChange);
        udB5Drop
                = (int) ((this.ud[B5][i - 1] - this.ud[B5][i]) / suddenChange);
        udB4Hike
                = (int) ((this.ud[B4][i] - this.ud[B4][i - 1]) / suddenChange);
        udDropNum = udAllDrop > udB5Drop ? udAllDrop : udB5Drop;
        udDropNum = udDropNum < udB4Hike ? udB4Hike : udDropNum;
        udDropNum *= udDropNum;
        if (udDropNum > 16) {
          udDropNum = 16;
        }
        this.agIndicator += udDropNum;
      }

      // Non-consective low and high UD vertices
      // If there is a sudden change in UD values in udist composite between
      // this vertex and its neighboring years, calculate the number of 
      // drops (or hikes) as before.  Square this value and add to 
      // this.agIndicator
      if (this.cstSeg[i] == NCLUD
              && this.ud[COMP][i - 1] - this.ud[COMP][i] > suddenChange
              && this.ud[COMP][i + 1] - this.ud[COMP][i] > suddenChange) {
        udDip = (int) (((this.ud[COMP][i - 1] + this.ud[COMP][i + 1]) / 2.0
                - this.ud[COMP][i]) / suddenChange);
        udDip *= udDip;
        if (udDip > 16) {
          udDip = 16;
        }
        this.agIndicator += udDip;
      }

      if (this.cstSeg[i] == NCHUD
              && this.ud[COMP][i] - this.ud[COMP][i - 1] > suddenChange
              && this.ud[COMP][i] - this.ud[COMP][i + 1] > suddenChange) {
        udHike = (int) ((this.ud[COMP][i] - (this.ud[COMP][i - 1]
                + this.ud[COMP][i + 1]) / 2.0) / suddenChange);
        udHike *= udHike;
        if (udHike > 16) {
          udHike = 16;
        }
        this.agIndicator += udHike;
      }
    }
  }

  /**
   * Smoooth out variability in initial determination of this pixel's high/low
   * ud segments.  This will remove NCLUD and NCHUD labels based on neighboring
   * values
   * 
   * @param smooth - Array in which to store smoothed segments calls
   * @param numYears - Number of years over which to run the smoothing
   * @param current - Current label to override
   */
  private void smoothSegment(byte[] smooth, int numYears, int current) {

    // Filter out non-consecutive low UD (NCLUD) and non-consecutive
    // high UD (NCHUD) vertex labels based on neighboring years'
    // information
    int i, j;
    byte newType, borderType;

    // Based on the vertex label (current) we're searching to change,
    // specify the new vertex label to assign as well as the border
    // condition label to find
    if (current == NCLUD) {
      newType = CHUD;
      borderType = NCHUD;
    } else {
      newType = CLUD;
      borderType = NCLUD;
    }

    // Search forward through the time series
    i = 1;
    while (i < numYears - 1) {
      // Skip if this isn't the target label
      if (smooth[i] != current) {
        i++;
        continue;
      }

      // If current label is between two new labels, change to newType
      if (smooth[i - 1] == newType && smooth[i + 1] == newType) {
        smooth[i] = newType;
        i++;
        continue;
      }

      // If current label is between a new label and a border label, change
      // to new and search forward for more border labels, changing them to 
      // newType if found
      if (smooth[i - 1] == newType && smooth[i + 1] == borderType) {
        smooth[i] = newType;
        j = i + 1;
        while (j < numYears && smooth[j] == borderType) {
          smooth[j] = newType;
          j += 1;
        }
        i = j;
      } // Current label was found but adjacent labels didn't qualify it for
      // a change
      else {
        i++;
      }
    }

    // Now, search backward through the time series, same logic as
    // above but in reverse.  There is no check for the "sandwich"
    // logic above; they have all been changed by this point
    i = numYears - 2;
    while (i > 0) {
      if (smooth[i] != current) {
        i--;
        continue;
      }
      if (smooth[i + 1] == newType && smooth[i - 1] == borderType) {
        smooth[i] = newType;
        j = i - 1;
        while (j >= 0 && smooth[j] == borderType) {
          smooth[j] = newType;
          j--;
        }
        i = j;
      } else {
        i--;
      }
    }
  }
  
  /**
   * Set disturbance information for a given time range
   * 
   * @param startYear - disturbance onset year
   * @param endYear  - disturbance finish year
   */
  private void setDisturbance(int startYear, int endYear) {
    // Set the disturbance year, disturbance flag and landcover type
    this.distYear[this.numDist] = startYear;
    this.distFlag[startYear] = JUST_DISTURBED;
    this.lcType = PART_FOREST;

    // Find the local UD peak within this disturbance segment and label
    // all intermediary years as transition, year is stored in true_peak,
    // ud is stored in maxUd 
    int i;
    int truePeak = startYear;
    double maxUd = this.ud[COMP][startYear];
    if (endYear >= this.numYears) {
      endYear = this.numYears - 1;
    }
    for (i = startYear + 1; i <= endYear; i++) {
      this.distFlag[i] = TRANSITION_PERIOD;
      if (this.ud[COMP][i] > maxUd && i < endYear
              && (this.ud[COMP][i] - this.ud[COMP][i - 1] < 2.0
              || this.ud[COMP][i] - this.ud[COMP][i + 1] < 2.0)) {
        maxUd = this.ud[COMP][i];
        truePeak = i;
      }
    }

    // Calculate the change magnitudes based on the peak year
    calculateChangeMagn(startYear, truePeak);

    // Fit a recovery (regrowth) regression line (B5 vs. years) from the
    // peak year to the end year. High goodness of fit value indicates
    // a very likely change
    double[] coeff;
    int distIndex = this.numDist;
    coeff = calculateLinearChangeRate(truePeak, endYear);
    this.regrR2[distIndex] = coeff[2];
    this.regrSlope[distIndex] = coeff[0];
    this.regrRough[distIndex]
            = fiRoughness(this.ud[COMP], startYear, endYear);

    // Fit a regression for the entire disturbance period
    coeff = calculateLinearChangeRate(startYear, endYear);
    this.distR2[distIndex] = coeff[2];
    this.distLength[distIndex] = (byte) (endYear - startYear + 1);

    // Increment the disturbance index
    this.numDist += 1;
  }

  /**
   * Set post-disturbance information for a given time range
   * 
   * @param startYear - post-disturbance onset year
   * @param endYear  - post-disturbance finish year
   */
  private void setPostDisturbForest(int startYear, int endYear) {
    // Mark all years in this segment as INTER_DISTURB_FOREST
    for (int i = startYear; i <= endYear; i++) {
      if (i >= this.numYears) {
        break;
      }
      this.distFlag[i] = INTER_DISTURB_FOREST;
    }
  }

  /**
   * Search for minor disturbances within a given time range
   * 
   * @param startYear - year to begin checking for disturbances
   * @param endYear - year to finish checking for disturbances
   */
  private void searchMinorDisturbances(int startYear, int endYear) {
    // Skip short duration disturbances
    if (endYear < startYear + 3) {
      return;
    }

    int i = startYear;
    int j;
    while (i <= endYear) {
      // Continue over bad pixels or no detected disturbances
      if (this.qFlag[i] == QA_BAD || isMinorFirst(i) == false) {
        i++;
        continue;
      }

      // Increment j as long as minor disturbance is detected
      // (is_minor_rest == 1).  If the year range is at least two years,
      // characterize it as a disturbance
      j = i + 1;
      while (j <= endYear && this.qFlag[j] == QA_GOOD
              && isMinorRest(j) == true) {
        j++;
      }
      if (j - i > 1) {
        setDisturbance(i, j);
      }
      i = j;
    }
  }

  /**
   * Calculate the change magnitude (in many indexes) for a given disturbance
   * segment
   * 
   * @param changeYear - year of initial disturbance onset
   * @param peakYear - year of peak disturbance
   */
  private void calculateChangeMagn(int changeYear, int peakYear) {
    // Handle exception
    // if (changeYear < 0 || changeYear >= this.numYears
    //         || peakYear < 0 || peakYear >= this.numYears) {
    //   String msg = "Year out of range in calculateChangeMagn";
    //   throw new Exception(msg);
    // }

    // Capture the disturbance magnitudes in UD, NDVI and DNBR spaces.
    // Magnitudes are relative from the peak UD year to the mean of
    // the pixel across all years
    this.distMagn[changeYear]
            = this.ud[COMP][peakYear] - this.meanForUdBx[COMP];
    this.distMagnB4[changeYear]
            = this.ud[B4][peakYear] - this.meanForUdBx[B4];
    this.distMagnVi[changeYear]
            = this.ud[NDVI][peakYear] - this.meanForUdBx[NDVI];
    this.distMagnBr[changeYear]
            = this.ud[DNBR][peakYear] - this.meanForUdBx[DNBR];
  }

  /**
   * Calculate a linear change rate in ud B5 over a given time span using a
   * simple linear fit
   * 
   * @param startYear - start year
   * @param endYear - end year
   * @return - Regression coefficients from least squares fit
   */
  private double[] calculateLinearChangeRate(int startYear, int endYear) {
    // Too few observations for meaningful regression, or not much
    // variability. Return default values
    if (endYear - startYear < 4 || this.fiRange < 0.1) {
      return new double[]{0.0, 25.0, 0.0, 0.0};
    }

    // Index errors on start_year, end_year
    if (endYear > this.numYears || startYear < 0) {
      // Handle exception
      // String msg = "Error: index error in calculateLinearChangeRate";
      // throw new Exception(msg);
    }

    if (endYear == this.numYears) {
      endYear--;
    }

    // Linear regression of B5 (Y) on years since disturbance (X)
    // TODO: Reimplement using LinearLeastSquares - for now using 
    // Cheng's code for consistency
    //
    // TODO: Generalize this function for any index
    int i, j;
    double[] x = new double[this.numYears];
    double[] y = new double[this.numYears];
    int numObs = endYear - startYear + 1;
    for (i = startYear; i <= endYear; i++) {
      j = i - startYear;
      y[j] = this.ud[B5][i];
      x[j] = this.yearTable[i] - this.yearTable[startYear];
    }
    return linearFit(y, x, numObs);
  }

  /**
   * Determine if a current year's pixel should be considered a minor
   * disturbance immediately after a disturbance event
   * 
   * @param curr - current index to check
   * @return - minor disturbance status
   */
  private boolean isMinorFirst(int curr) {
    // TODO: These thresholds are getting recalculated each time even
    // though they are not dependent on the value of curr.  Needs to be
    // calculated once per pixel and passed

    // Define change thresholds for UD, UD_B5, NDVI and DNBR based on
    // mean, sd values
    double chgThrUd, chgThrDnbr, chgThrNdvi, chgThrB5;
    chgThrUd = this.meanForUdBx[COMP] + 1.5 + this.sdForUdBx[COMP];
    chgThrDnbr = this.meanForUdBx[DNBR] - 0.15 - this.sdForUdBx[DNBR];
    chgThrNdvi = this.meanForUdBx[NDVI] - 0.15 - this.sdForUdBx[NDVI];
    chgThrB5 = this.meanForUdBx[B5] + 1.0 + this.sdForUdBx[B5];

    // First year of a segment
    if (curr == 0) {
      if ((this.ud[COMP][curr] > chgThrUd + 1.0
              || this.ud[B5][curr] > chgThrB5 + 1.0)
              && this.ud[NDVI][curr] < this.meanForUdBx[NDVI]
              && this.ud[DNBR][curr] < this.meanForUdBx[DNBR]
              || (this.ud[NDVI][curr] < chgThrNdvi - 0.1
              || this.ud[DNBR][curr] < chgThrDnbr - 0.1)) {
        return true;
      } else {
        return false;
      }
    } // Other years in the segment
    else {
      if ((((this.ud[COMP][curr] > chgThrUd
              || this.ud[B5][curr] > chgThrB5)
              && this.ud[NDVI][curr] < this.meanForUdBx[NDVI]
              && this.ud[DNBR][curr] < this.meanForUdBx[DNBR])
              || (this.ud[DNBR][curr] < chgThrDnbr
              || this.ud[NDVI][curr] < chgThrNdvi))
              && (this.ud[COMP][curr] > this.ud[COMP][curr - 1] + 2.0
              || this.ud[B5][curr] > this.ud[B5][curr - 1] + 2.0
              || this.ud[DNBR][curr] < this.ud[DNBR][curr - 1] - 0.2)) {
        return true;
      } else {
        return false;
      }
    }
  }

  /**
   * Determine if a current year's pixel should be considered a minor
   * disturbance following a disturbance event (but not immediately - different
   * logic for first year and other years after a disturbance event) 
   * 
   * @param curr - current index to check
   * @return - minor disturbance status
   */
    private boolean isMinorRest(int curr) {
    // Define change thresholds for UD, UD_B5, NDVI and DNBR based on
    // mean, sd values
    double chgThrUd, chgThrDnbr, chgThrNdvi, chgThrB5;
    chgThrUd = this.meanForUdBx[COMP] + 1.0 + this.sdForUdBx[COMP] / 2.0;
    chgThrDnbr = this.meanForUdBx[DNBR] - 0.1 - this.sdForUdBx[DNBR];
    chgThrNdvi = this.meanForUdBx[DNBR] - 0.1 - this.sdForUdBx[NDVI];
    chgThrB5 = this.meanForUdBx[B5] + 1.0 + this.sdForUdBx[B5] / 2.0;

    if (((this.ud[COMP][curr] > chgThrUd
            || this.ud[B5][curr] > chgThrB5)
            && (this.ud[NDVI][curr] < this.meanForUdBx[NDVI]
            || this.ud[DNBR][curr] < this.meanForUdBx[DNBR]))
            || (this.ud[DNBR][curr] < chgThrDnbr
            || this.ud[NDVI][curr] < chgThrNdvi)) {
      return true;
    } else {
      return false;
    }
  }

  /**
   * Class for retaining information about segments across a time series
   */
  public class TSSegment {

    public int totSeg;             // total number of segments for this pixel
    public byte[] segType;         // type for each segment
    public byte[] segLength;       // length for each segment
    public byte[] startYear;       // start year for each segment
    public byte[] endYear;         // end year for each segment
    public byte[] peakYear;        // peak yeer for each segment
    public double[] lfSlp;         // linear-fit slope for each segment
    public double[] lfR2;          // linear-fit R2 for each segment

    // TODO: We are making this numYears long when really it is likely much 
    // shorter.  We could trim once we obtain the segments
    public TSSegment(int numYears) {
      this.totSeg = 0;
      this.segType = new byte[numYears];
      this.segLength = new byte[numYears];
      this.startYear = new byte[numYears];
      this.endYear = new byte[numYears];
      this.peakYear = new byte[numYears];
      this.lfSlp = new double[numYears];
      this.lfR2 = new double[numYears];
    }
  }

  // Almost certainly, these are pre-existing functions somewhere???
  // No error checking done here ...
  
  /**
   * Get a mean value over a slice of an array
   * 
   * @param arr - Array to calculate mean
   * @param start - start index of array
   * @param end - end index of array  (not included!)
   * @return - mean value of slice
   */
  private double getSliceMean(double[] arr, int start, int end) {
    double sum = 0.0;
    for (int i = start; i < end; i++) {
      sum += arr[i];
    }
    return ((double) sum / (end - start));
  }

  /**
   * Get a standard deviation value over a slice of an array
   * 
   * @param arr - Array to calculate standard deviation
   * @param start - start index of array
   * @param end - end index of array  (not included!)
   * @return - standard deviation value of slice
   */
  private double getSliceStd(double[] arr, int start, int end) {
    double var = 0.0;
    double mean = getSliceMean(arr, start, end);
    for (int i = start; i < end; i++) {
      var += (arr[i] - mean) * (arr[i] - mean);
    }
    return Math.sqrt((double) var / (end - start));
  }

  /**
   * Least-squares linear fit of Y on X
   * 
   * @param y - array of dependent values
   * @param x - array of independent values
   * @param numObs - length of array
   * @return - slope, intercept, r2 and t value of least-squares linear regr.
   */
  private double[] linearFit(double[] y, double[] x, int numObs) {
    // TODO: Probably should be replaced with generic linear fitting algorithm
    // but kept in here for comparison purposes

    int i;
    double meanX, meanY, slope, intercept, r2, denom, t;
    double sxx = 0.0;
    double sxy = 0.0;
    double syy = 0.0;

    meanX = getSliceMean(x, 0, numObs);
    meanY = getSliceMean(y, 0, numObs);
    for (i = 0; i < numObs; i++) {
      sxx += (x[i] - meanX) * (x[i] - meanX);
      syy += (y[i] - meanY) * (y[i] - meanY);
      sxy += (y[i] - meanY) * (x[i] - meanX);
    }

    if (sxx < 0.00001) {
      // Handle exception
      // String msg = "Error: divided by 0 in linear fit\n";
      // throw new Exception(msg);
    }

    slope = sxy / sxx;
    intercept = meanY - meanX * slope;
    if (syy < 0.00001) {
      r2 = 0.0;
    } else {
      r2 = (sxy * sxy) / (sxx * syy);
    }
    denom = r2 == 1.0 ? 0.00001 : 1.0 - r2;

    // TODO: What is t?
    t = Math.sqrt(r2 * (numObs - 2) / denom);

    return new double[]{slope, intercept, r2, t};
  }

}
