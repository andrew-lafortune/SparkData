### Execution
To execute this project, run COVIDTrafficDataPreparation first, then COVIDTrafficDataModeling. Both can be found published on Databricks community here:
1. [COVIDTrafficDataPreparation](https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/1870284974891931/647730006675301/238308696242763/latest.html)
2. [COVIDTrafficDataModeling](https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/1870284974891931/3804833370157229/238308696242763/latest.html)

### Background
These notebooks explore traffic data recorded by Weigh In Motion (WIM) stations around the state in the context of the COVID 19 pandemic as a way of measuring the effectiveness of various policy changes such as the closure of public entertainment venues and the gradual reopening of businesses to partial capacity as the rates of infection and death decreased. The main deliverable of the project is the graph below showing the traffic volume during the pandemic relative to pre-pandemic traffic volume collected from 2017-2019 aligned with graphs of total COVID infections and deaths. The black vertical lines on each plot represent preventative measures like stay-at-home orders, and blue lines represent relaxation measures like the reopening of schools. All of the raw data used to create the graphs is available in the `data` folder, and more information on the composition of data can be seen in the `COVIDTrafficDataPreparation` notebook.

### Final Timeline Graph
![Full Traffic, Infection, Death Timeline](/img/FinalTimeline.png)
