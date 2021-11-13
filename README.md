# Climate Change Solutions - EV Branch
<b>Assumptions:</b>\
-The values for how much light vehicles contribute to the US emissions are constant.\
-The averages for carbon emitted by cars is also the median or at least very close to it.\
-No new emissions reduction legislation is passed over the span of the entire forecasting period.\
-No new technology to reduce light vehicle emissions gets invented over the span of the entire forecasting period.\
-No new trends will occur outside of growing EV adoption(ie. median price of EVs drop significantly) that massively disrupts the car market over the span of the entire forecasting period.\
-The current car lineup remains the same in terms of the ratio of EV to non-EV offerings.\
-EV sales do not follow the S curve.\
-Plug-in hybrid owners do not use the battery at all and therefore emit as much carbon as ICE vehicles.\
-The American powergrid can handle an infinite number of EVs charging.

<b>Limitations:</b>\
-Since EVs are relatively new technology, there is not enough historical data to create a solid normal distribution for the ARIMA model in some cases.\
-As further consequence, the potential values for p, d, and q are limited.\
-Currently, I cannot find data for carbon emissions that are seasonal.

<b>Current plan</b>:\
-~~Forecast EV sales to check if yearly sales will reach, at worst, half of all light vehicle sales by 2030.~~ Done.\
-~~If this is the case, assume this means that the forecasted value of registered EV vehicles would be the baseline for the solution.~~ Done.\
-~~If this is not the case, figure out how much sales need to grow in order to meet the goal and determine how many registered EVs will be on the road if this were to happen. There will then be two forecast value of EV registrations, a theoretical baseline and the forecasted values.~~ Done.\
-~~Subtract the average carbon emissions of light vehicles times registered EVs from the forecasted values of carbon emissions in the US. If a theoretical baseline is needed, do it for that as well.~~ Done.

<b>Optional steps if time permits:</b>\
-Determine the average carbon emissions by cars based on registration data relative to US emissions instead of a value found online.\
-Perform forecasting on various thresholds either based on EV Sales market share or EV cars registration share.\
-Consider using vars since EV car sales and EV car registrations are linked together due to being fairly new technology.

<b>To do:</b>\
-Polish code and datasets then recollect images.\
-~~Collect images.~~ Done.\
-~~Clean branch.~~ Done.
