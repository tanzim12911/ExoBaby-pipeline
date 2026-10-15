================================================================================
IMPORTANT NOTE REGARDING FIGURE HEADINGS / PLOT TITLES
================================================================================

Some generated plot figures (.png and .pdf files found inside the various 
'analysis/.../figures/' subdirectories) may contain static or generic figure 
headings, titles, or axis labels (for example, referencing "valid129" or standard 
English CDI labels) even when displaying data for other benchmarks (such as 
BanglaProtha L2/L2', Persian CDI, Turkish CDI, or 284 CDI).

This is solely due to a minor hardcoded title template in our automated figure 
generation pipeline script during batch rendering. 

Please be assured that:
1. The underlying numerical results, CSV data files, JSON summaries, and frame-level 
   detections in all 'results/' and 'frame_data/' directories are EXACT and ACCURATE 
   for their respective specific benchmark datasets.
2. The data points plotted on the curves, bars, and log-log distribution charts 
   accurately reflect the actual experiment for that folder. Only the rendered text 
   heading template string remained fixed across batch runs.


Sub-directory structure:


<ExoBaby-results Folder>/
  ├── analysis/
  │   ├── figures/    --> distribution plots, log-log plots, bar charts (.png, .pdf)
  │   └── results/    --> quantitative summary tables, CSV metrics, OOI (Objects of Interest) coverage JSONs
  ├── data/           --> CDI category text lists, word lists(If this folder is empty then CDI collected from GitHub repo)
  └── frame_data/     --> Full frame-level detection CSVs containing bounding boxes, labels, 
                          confidence scores and frame metadata


Detailed Directory Map & Contents:

1. Detection_Results(frame_data)
   - Stores raw and merged union frame detection results across experiments where 
     full pipeline analysis was not re-run (e.g., union of English, Bangla, Persian, 
     and Turkish CDIs, Qwen 9B outputs, YOLO+CLIP 500-frame subsets).
   - Contains a specific Readme.txt detailing post-processed union detection files.

2. ExoBaby-results(BD 10524 frames on Gemini_filtered_284_English_CDI)
   - Results for 10,524 Bangladeshi (BD) video frames evaluated against an 
     284 English CDI category set filtered by Gemini.
   - Includes full analysis figures, results metrics, CDI data, and filtered frame detections.

3. ExoBaby-results(BD 10524 frames on Paper 129 CDI)
   - Benchmark results for 10,524 Bangladeshi (BD) video frames evaluated on the 
     standard paper baseline of 129 CDI categories.

4. ExoBaby-results(BanglaProtha_L2')
   - Bangla language visual CDI evaluation under the L2' (disjoint category) setup.
   - Contains distribution figures, provenance CSVs, and frame detection metadata.

5. ExoBaby-results(BanglaProtha_L2)
   - Bangla language visual CDI evaluation under the L2 (pre-disjoint category) setup.

6. ExoBaby-results(English 2259 +10023 frames)
   - Combined English video frame analysis across 2,259 initial frames + 10,023 frames.
   - Contains long-tailed bar charts, log-log plots, and domain breakdown statistics.

7. ExoBaby-results(English first 10023 frames on paper 129 CDI)
   - Detailed evaluation of the first 10,023 English frames against the 129 paper CDI list.
   - Includes model stability test comparisons, CLIP validation reports, and frequency rank 
     correlation sheets against the BabyView dataset.

8. Persian_Results
   - Contains 'ExoBaby-results(Persian1)': Evaluation of Persian language CDI categories 
     applied to Bangladeshi (BD) video frames.

9. Turkish_Results
   - Contains 'ExoBaby-results(Turkish1)': Evaluation of Turkish language CDI categories 
     applied to Bangladeshi (BD) video frames.

10. Pie_Chart_(L1+L2')_On BD dataset
    - High-resolution grid images and comparative pie charts visualizing category breakdowns 
      and priority views (English vs Bangla priority, 4-way grid views) for L1 and L2'.

11. Qwen_RawYOLO_YOLO+CLIP Comparison
    - Model performance comparison between Qwen (open-vocabulary), Raw YOLO, and YOLO+CLIP.
    - Includes 'qwen_vs_yolo_3way_comparison.csv' and 50 side-by-side annotated visualization 
      frame images in 'side_by_side_visualizations/'.

12. Visual Comparison of Qwen+Clip, YOLO, YOLO+Clip
    - Dedicated visual comparison directory containing side-by-side image samples (50 frames) 
      demarcating bounding box detections and model outputs for Qwen+CLIP vs YOLO vs YOLO+CLIP.

13. RandomUS_Baseline_Results(Not specific to babyview)
    - Control baseline experiment using 10,023 random US vlog frames (V2_RandomUS10K).
    - Contains 'exobaby_vs_random_us_summary_report.md' detailing power-law exponents, 
      Jaccard similarities, Spearman rank correlations, and top-15 category overlaps.

14. ExoBaby-results (English 12,282 frames on L1+L2')
    - Evaluation of L1 + L2' (English + Turkish + Persian + BanglaProtha Disjoint) across 
      the 12,282 English video frames dataset.
    - Another pie chart folder is also added of language wise break down of category and object detection.


Key Data Metrics:

- merged_frame_detections_with_metadata_filtered-0.27.csv:
  The primary output CSV file in each 'frame_data/' folder, listing every detected 
  object instance per frame.

- ooi_coverage.json:
  Summary JSON documenting Objects of Interest (OOI) coverage and category statistics.

- long_tailed_dist_prop_included_categories_filtered-0.27_valid129.csv:
  Proportional counts and probability distributions of categories in long-tailed visual distributions.

- fig1a_long_tailed_bar / fig1b_loglog / fig2_per_domain / fig3_provenance_top5:
  Standard set of distribution visualizations generated for each experiment run.

