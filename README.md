# learn locations
This is a start for me to touch skilearn! This package has been made super powerful!
==========================================================================================
The task is to do regression on real locations and classification on building ID and floor number on the UJIIndoorLoc database. Just play around this data, you could learn a lot about machine learning, such as dimensional reductions, data normalization, so so more! 

 Flow 
 ==========
Delete NANs → scipy.stats.boxcox(reduce the positive skewness) → PCA(reduce dimensions and form linearly uncorrelated variables) → Gridsearch(parameters tuning within several methods) → choose the best estimator ∗[lat, lon by RandomForests, buildingID by RandomForests]

More
=====
I will update this later for beginners in skilearn and basic machine learning to show what I have learned based this data!


References

[1]. Nguyen, Khuong An. ”A performance guaranteed indoor positioning system using conformal prediction and the WiFi signal strength.” Journal of Information and Telecommunication 1.1 (2017): 41-65.

[2]. Akram, Beenish A., Ali H. Akbar, and Omair Shafiq. ”HybLoc: Hybrid indoor Wi-Fi localization using soft clustering-based random decision forest en- sembles.” IEEE Access 6 (2018): 38251-38272.

[3]. Caroline Katba, ”Predicting Indoor Location Using WiFi Fingerprinting, by https://katba-caroline.com/predicting-indoor-location-using-wifi-fingerprinting/, March 27, 2019.

[4]. Torres-Sospedra, Joaqun, et al. ”UJIIndoorLoc: A new multi-building and multi-floor database for WLAN fingerprint-based indoor localization problems.” 2014 international conference on indoor positioning and indoor navigation (IPIN). IEEE, 2014.
