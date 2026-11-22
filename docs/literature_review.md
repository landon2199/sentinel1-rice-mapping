# Literature review: mapping rice (and rice vs corn) with Sentinel-1 and Sentinel-2

This is a short review written for this project. Each paper below was checked against its DOI record (Crossref). The notes give only what the abstract or paper states. "OA" means overall accuracy.

## 1. Why radar sees rice: the flooding signal

Radar backscatter is the share of the radar pulse that bounces back to the satellite. Calm water acts like a mirror. It reflects the pulse away, so a flooded field looks dark. As rice plants grow out of the water, the pulse bounces between the water and the stems and scatters inside the canopy, so the field gets brighter. This dark-then-bright pattern is the main radar signature of paddy rice. VH (sent vertical, received horizontal) is mostly scattered by the canopy volume, so it tracks plant growth closely. VV (vertical in and out) reacts more to the soil and water surface.

- **Nguyen, Gruber & Wagner (2016)**, Mekong Delta, Vietnam. Sentinel-1A VV and VH, October 2015 to March 2016. VH was more sensitive to rice growth than VV. They took phenology dates (start of season, heading, season length) from the VH time series and used a decision tree, not machine learning. Checked against a 2015 land-use map: OA 87.2 %, kappa 0.71. Provincial rice areas agreed with official statistics (R² = 0.98).
- **Bazzi et al. (2019)**, Camargue, France. Sentinel-1 time series over 832 plots of several crop types. Features were fitted Gaussian-curve metrics and the variance of the VV/VH series, plus the slope of the VH series. A decision tree (OA 96.3 %) and Random Forest (OA 96.6 %) performed about the same. Validated against national plot data.
- **Clauss, Ottinger & Kuenzer (2018)**, six rice regions worldwide. They averaged the Sentinel-1 time series over superpixels (small image segments), which reduces speckle. Average OA was 0.83.
- **Nelson et al. (2014)**, RIICE project, 13 sites in 6 Asian countries. This used X-band radar (HH), not Sentinel-1. A rule-based method used the rice backscatter time signature. More than 1,300 field observations gave 85 to 95 % accuracy. The authors note that rainy-season cloud rules out optical imagery.

## 2. Phenology rules vs machine learning

- **Dong & Xiao (2016)** review paddy-rice mapping methods in four groups: (1) reflectance and image statistics; (2) vegetation indices plus image statistics; (3) time-series analysis of vegetation indices or radar backscatter; (4) phenology-based detection of key growth phases. In optical data, the flooding/transplanting signal is when the land surface water index (LSWI) briefly rises above NDVI or EVI.
- **Xiao et al. (2005)** built that rule for MODIS (500 m, 8-day) over southern China. LSWI, EVI and NDVI time series find the flooding and transplanting period. **Xiao et al. (2006)** extended it to South and Southeast Asia. A pixel counts as flooded when LSWI + 0.05 ≥ EVI or NDVI. Rice growth is then confirmed if EVI reaches half its maximum within 40 days. Areas were compared with national and subnational statistics.
- **Dong et al. (2016)** ran a phenology-based flooding algorithm on about 3,290 Landsat 8 scenes in Google Earth Engine for Japan, the Koreas and northeast China. Validated with very-high-resolution images and field photos: producer's accuracy 73 %, user's accuracy 92 %.
- **Torbick et al. (2017)**, Myanmar, 632 Sentinel-1A scenes. A Random Forest land-cover map (Sentinel-1 + Landsat 8 + PALSAR-2) had mean out-of-sample kappa above 0.90. The rice mask itself came from Sentinel-1 phenology indicators (dynamic range, inundation, growth stages). Harvested area against the government census: R² = 0.78.
- **Singha et al. (2019)**, Bangladesh and northeast India. 10 m Sentinel-1 with Random Forest in Google Earth Engine for three rice seasons (Boro, Aus, Aman). Accuracy was above 90 % on independent validation samples. They also compared their maps with IRRI and MODIS rice maps.

Rule-based methods need no training labels, but their thresholds assume a flood at transplanting. Machine-learning methods learn the signature from labels, so they can fit other practices (such as a delayed flood), but only where labels exist.

## 3. Rice vs maize/corn, and US studies with CDL labels

- **Ferrant et al. (2017)**, Kudaliar catchment, Telangana, South India (about 1,000 km²). Sentinel-1 and Sentinel-2 with Random Forest, 2016 to 2017. Classes were inundated rice paddy, maize, other irrigated crops and water. The authors note that Sentinel-1 gives usable images in the cloudy monsoon season. They report "moderate classification confusion"; the abstract gives no single OA.
- **Verma, Kumar & Lal (2019)** combined Sentinel-1 SAR and Sentinel-2 optical data for kharif (monsoon-season) crops in India. Only the bibliographic record was checked; the paper's details were not read for this review.
- **Wei et al. (2022)** used CDL as labels for a Sentinel-1 time series in Arkansas to train a U-Net (a deep-learning segmentation model). They then transferred it to Jiangsu, China, with pseudo-labels from K-means and Random Forest.
- **Lin et al. (2022)** mapped rice across the major US rice areas from Sentinel-1 alone. A multi-task LSTM (a deep-learning time-series model) reached OA 98.3 % and F1 0.804 and did better than Random Forest. Low VH early in the season and high VH later were the key rice cues. The label source and validation design were not confirmed for this review.

## 4. Why radar matters: clouds

- **Whitcraft et al. (2015)** used 10 to 13 years of twice-daily MODIS cloud flags over the world's cropland. Cloud is frequent and widespread early and mid-season, which is when crop-type mapping needs images. In many areas fewer than half of 8-day composites are even 70 % clear.

## 5. Transplanted rice in Asia vs drill-seeded, delayed-flood rice in Texas

- In Asia, rice is commonly grown by transplanting seedlings into puddled (wet-tilled) soil (Kumar & Ladha 2011). The field is flooded at planting, which gives the strong dark-water start that sections 1 and 2 rely on.
- Texas guidelines describe seeding into dry soil (drilled, or broadcast and then flushed) or water-seeding. The **permanent flood is set when seedlings reach the 4 to 5-leaf or second-tiller stage** (Texas Rice Production Guidelines, AgriLife, 2012 edition). Optimum planting in the western rice area is about 15 March to 21 April. Ratoon cropping is common: after the main harvest the stubble is fertilized and reflooded to grow a second crop. Between 26 % and 58 % of Texas main-crop acres were ratooned each year from 1993 to 2011 (same guidelines, Table 40). Main-crop harvest is typically finished by mid-August (AgriLife Today, 3 August 2021).
- So in a drill-seeded Texas field, radar first sees dry or moist bare soil, not open water. The flood arrives when plants are already a few leaves tall. The deepest "dark water" dip of transplanted rice should therefore be weaker or absent, and a second growth cycle (the ratoon) may appear after the August harvest. This project checks that pattern in the data instead of assuming it (see the class curves in the README).

## 6. How the published accuracies were validated

Most studies above report accuracy against a land-use map, field points, VHR image interpretation or official area statistics. None of the abstracts read here describes spatially separated (block) validation, and several do not say how training and test samples were separated. When training and test pixels come from the same fields, accuracy is usually optimistic (Roberts et al. 2017). The numbers are therefore not directly comparable with each other or with this project.

## 7. What this project follows and where it differs

- **Follows:** per-pixel Random Forest on Sentinel-1 VV, VH and VH/VV time series (as in Singha et al. 2019 and the Random Forest runs of Bazzi et al. 2019 and Torbick et al. 2017), with Sentinel-1 + Sentinel-2 combined as in Ferrant et al. (2017). Labels come from the USDA CDL, as in Wei et al. (2022).
- **Differs:**
  1. It tests a drill-seeded, delayed-flood rice system with a ratoon crop, not transplanted paddy.
  2. It separates rice from corn as well as from everything else.
  3. It compares S1-only, S2-only and S1+S2 models on the same samples, including an in-season test of how early each sensor works.
  4. It validates on held-out 2 km blocks with a buffer, compares that with random cross-validation, and estimates areas with confidence intervals (Olofsson et al. 2014), instead of reporting one overall accuracy.
  5. It uses no phenology thresholds, because a rule tuned to transplanting floods would not fit the Texas calendar.

## References

- Bazzi, H., Baghdadi, N., El Hajj, M., Zribi, M., Minh, D. H. T., Ndikumana, E., Courault, D., & Belhouchette, H. (2019). Mapping paddy rice using Sentinel-1 SAR time series in Camargue, France. *Remote Sensing*, 11(7), 887. https://doi.org/10.3390/rs11070887
- Clauss, K., Ottinger, M., & Kuenzer, C. (2018). Mapping rice areas with Sentinel-1 time series and superpixel segmentation. *International Journal of Remote Sensing*, 39(5), 1399–1420. https://doi.org/10.1080/01431161.2017.1404162
- Dong, J., & Xiao, X. (2016). Evolution of regional to global paddy rice mapping methods: A review. *ISPRS Journal of Photogrammetry and Remote Sensing*, 119, 214–227. https://doi.org/10.1016/j.isprsjprs.2016.05.010
- Dong, J., Xiao, X., Menarguez, M. A., Zhang, G., Qin, Y., Thau, D., Biradar, C., & Moore, B. (2016). Mapping paddy rice planting area in northeastern Asia with Landsat 8 images, phenology-based algorithm and Google Earth Engine. *Remote Sensing of Environment*, 185, 142–154. https://doi.org/10.1016/j.rse.2016.02.016
- Ferrant, S., Selles, A., Le Page, M., et al. (2017). Detection of irrigated crops from Sentinel-1 and Sentinel-2 data to estimate seasonal groundwater use in South India. *Remote Sensing*, 9(11), 1119. https://doi.org/10.3390/rs9111119
- Kumar, V., & Ladha, J. K. (2011). Direct seeding of rice: Recent developments and future research needs. *Advances in Agronomy*, 111, 297–413. https://doi.org/10.1016/B978-0-12-387689-8.00001-1
- Lin, Z., Zhong, R., Xiong, X., et al. (2022). Large-scale rice mapping using multi-task spatiotemporal deep learning and Sentinel-1 SAR time series. *Remote Sensing*, 14(3), 699. https://doi.org/10.3390/rs14030699
- Nelson, A., Setiyono, T., Rala, A. B., et al. (2014). Towards an operational SAR-based rice monitoring system in Asia: Examples from 13 demonstration sites across Asia in the RIICE project. *Remote Sensing*, 6(11), 10773–10812. https://doi.org/10.3390/rs61110773
- Nguyen, D. B., Gruber, A., & Wagner, W. (2016). Mapping rice extent and cropping scheme in the Mekong Delta using Sentinel-1A data. *Remote Sensing Letters*, 7(12), 1209–1218. https://doi.org/10.1080/2150704X.2016.1225172
- Olofsson, P., Foody, G. M., Herold, M., Stehman, S. V., Woodcock, C. E., & Wulder, M. A. (2014). Good practices for estimating area and assessing accuracy of land change. *Remote Sensing of Environment*, 148, 42–57. https://doi.org/10.1016/j.rse.2014.02.015
- Roberts, D. R., Bahn, V., Ciuti, S., et al. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography*, 40(8), 913–929. https://doi.org/10.1111/ecog.02881
- Russell, A. (2021, August 3). Texas rice farmers enduring tough season. *AgriLife Today*, Texas A&M AgriLife. https://agrilifetoday.tamu.edu/2021/08/03/texas-rice-farmers-enduring-tough-season/
- Singha, M., Dong, J., Zhang, G., & Xiao, X. (2019). High resolution paddy rice maps in cloud-prone Bangladesh and Northeast India using Sentinel-1 data. *Scientific Data*, 6, 26. https://doi.org/10.1038/s41597-019-0036-3
- Texas A&M AgriLife (2012). *Texas Rice Production Guidelines*. https://varietytesting.tamu.edu/wp-content/uploads/sites/17/legacy-files/rice/varietyfiles/2012_Rice_Production_Guidelines.pdf
- Torbick, N., Chowdhury, D., Salas, W., & Qi, J. (2017). Monitoring rice agriculture across Myanmar using time series Sentinel-1 assisted by Landsat-8 and PALSAR-2. *Remote Sensing*, 9(2), 119. https://doi.org/10.3390/rs9020119
- Verma, A., Kumar, A., & Lal, K. (2019). Kharif crop characterization using combination of SAR and MSI Optical Sentinel Satellite datasets. *Journal of Earth System Science*, 128(8), 230. https://doi.org/10.1007/s12040-019-1260-0
- Wei, P., Huang, R., Lin, T., & Huang, J. (2022). Rice mapping in training sample shortage regions using a deep semantic segmentation model trained on pseudo-labels. *Remote Sensing*, 14(2), 328. https://doi.org/10.3390/rs14020328
- Whitcraft, A. K., Vermote, E. F., Becker-Reshef, I., & Justice, C. O. (2015). Cloud cover throughout the agricultural growing season: Impacts on passive optical earth observations. *Remote Sensing of Environment*, 156, 438–447. https://doi.org/10.1016/j.rse.2014.10.009
- Xiao, X., Boles, S., Liu, J., Zhuang, D., Frolking, S., Li, C., Salas, W., & Moore, B. (2005). Mapping paddy rice agriculture in southern China using multi-temporal MODIS images. *Remote Sensing of Environment*, 95(4), 480–492. https://doi.org/10.1016/j.rse.2004.12.009
- Xiao, X., Boles, S., Frolking, S., Li, C., Babu, J. Y., Salas, W., & Moore, B. (2006). Mapping paddy rice agriculture in South and Southeast Asia using multi-temporal MODIS images. *Remote Sensing of Environment*, 100(1), 95–113. https://doi.org/10.1016/j.rse.2005.10.004
