# Comparative Analysis: ExoBaby vs. Random US Vlogs (Valid Run)

This report compares the object distribution from **ExoBaby (10,023 frames)** against **Random US Vlogs (10,023 frames)** using the fixed YOLOE+CLIP detection (threshold 0.27) across 129 CDI categories.

## 1. High-Level Summary Statistics
| Metric | ExoBaby | Random US Vlogs |
| :--- | :---: | :---: |
| Total Frames | 10,023 | 10,023 |
| Frames with Detections | 4,889 (48.78%) | 4,842 (48.31%) |
| Detected Categories | 96 | 95 |
| Power-law Exponent (alpha) | 1.8701 | 1.5631 |

**Overall Spearman Rank Correlation:** **0.7457** (p = 3.135e-20)

## 2. Top Category Overlaps
* **Top-10 Categories:** 7 shared categories (Jaccard similarity: 53.85%)
  * Shared: bottle, chair, cup, door, lamp, necklace, pillow
* **Top-20 Categories:** 16 shared categories (Jaccard similarity: 66.67%)
  * Shared: bed, bottle, chair, couch, cup, door, glasses, hand, hat, lamp, mouth, necklace, pillow, plant, table, towel

## 3. Top 15 Categories Comparison Side-by-Side

| Rank | ExoBaby Category | ExoBaby Prop | Random US Category | Random US Prop |
| :---: | :--- | :---: | :--- | :---: |
| 1 | pillow | 9.0492% | necklace | 12.0822% |
| 2 | chair | 7.1635% | lamp | 5.8166% |
| 3 | necklace | 5.0085% | cup | 5.2878% |
| 4 | cup | 4.7291% | pillow | 5.2380% |
| 5 | plant | 3.9808% | bed | 3.6416% |
| 6 | bottle | 3.9409% | glasses | 2.9732% |
| 7 | couch | 3.8811% | bottle | 2.8534% |
| 8 | lamp | 3.7713% | mouth | 2.4943% |
| 9 | hat | 3.3623% | chair | 2.4943% |
| 10 | door | 3.0230% | door | 2.2349% |
| 11 | table | 2.6938% | hand | 2.0952% |
| 12 | bed | 2.6339% | plant | 1.9655% |
| 13 | refrigerator | 2.0453% | couch | 1.5664% |
| 14 | stroller | 1.2571% | box | 1.4766% |
| 15 | hand | 1.1972% | bowl | 1.4068% |

## 4. Key Takeaways
- **Yield Rate:** The proportion of frames containing objects of interest is almost identical between ExoBaby (48.78%) and Random US Vlogs (48.31%) now that YOLOE channels are fixed.
- **Distribution Shape:** Both datasets display a highly skewed, long-tailed power-law distribution. The exponent is slightly different (ExoBaby alpha = 1.87 vs. Random US alpha = 1.56), reflecting differences in composition.
- **Dominant Objects:** Both distributions are heavily dominated by indoor/household items like `pillow`, `necklace` (often high in open-vocabulary crop detection), `lamp`, `cup`, and `chair`. However, the exact order of ranks shows some variation.
