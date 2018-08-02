---
title: Chicago Divvy Bicycle
author: Ding Ma
layout: post
---

### Chicago Divvy Bicycle Project

[Divvy](https://www.divvybikes.com/) is a bicycle sharing system in the City of Chicago and two adjacent suburbs operated by Motivate for the Chicago Department of Transportation. It operates 5800 bicycles at 580 stations in an area bounded by 87th Street on the south, Central Street in Evanston on the north, Rainbow Beach Park near South Shore Drive on the east, and Harlem Avenue in Oak Park on the west.

In this blog, I proformed a series of data processing, visualization, and analysis. All the source code can be found in my Github page.

### Data
#### Bicycle Data
You can download the Divvy Bicycle data from 2013 to 2017 [here](https://www.divvybikes.com/system-data).
In this data, each trip is anonymized and includes:
  * Trip start day and time
  * Trip end day and time
  * Trip start station
  * Trip end station
  * Rider type (Member, Single Ride, and Explore Pass)
  * If a Member trip, it will also include Member’s gender and year of birth

#### Weather Data
In this project, I write a wrapper to download the historical weathers of Chicago between 2013-2017 from [National Oceanic and Atmospheric Administration](https://www.noaa.gov).

### Visualization
  * Here is an [interactive map](https://dingma129.github.io/assets/active_image/station_distribution.html) for station distribution. We can see that most of the stations are located in the downtown area.

  * Annual total trip counts were growing from 2013 to 2017. However, the average trip duration did not change at all.

<img src="https://dingma129.github.io/assets/figures/divvy/tripcountvsyear.png" width="800">

  * During June, July, August and September, there were more total trip counts. The average trip duration was also relatively longer during those four months. 

<img src="https://dingma129.github.io/assets/figures/divvy/tripcountvsmonth.png" width="800">

  * There were more rides during weekday comparing to weekend. But the average trip duration was longer during weekend. 

<img src="https://dingma129.github.io/assets/figures/divvy/tripcountvsdayofweek.png" width="800">

  * If we look into detail, we can see two totally different behaviors.
     * During weekday, there were two peaks. One happened at 7:00-8:59, and the other happened at 16:00-18:59.
     * During weekend, there was plenty of rides during 8:00-20:59 with no sharp peak.

<img src="https://dingma129.github.io/assets/figures/divvy/tripcountvshour.png" width="800">

### Analysis



### Conclusion