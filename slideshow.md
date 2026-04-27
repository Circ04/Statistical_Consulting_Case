Page
Page number
1
of 19
Tim
Hellemans
ORTEC
Forecasting @ ECG
Who am I professionaly?
Tim Hellemans
▪ Master in (Fundamental) mathematics,
▪ Phd Computer Science
▪ Data science consultant @ ORTEC
& Data science teacher @ Syntra AB
For fun I like to:
Fun mental
What is consultancy?
3
4
We also have a (very cool) office
in Amsterdam, where we do a
lot of the ECG work!
What do we offer?
5
We actively collaborate with universities on:
• Master theses,
• internships,
• Traineeships (first jobs!)
• …
https://ortec.com/en/careers/find-jobs
We are also working on a thesis topic within this project!
Problem at hand
Scope:
>5000 employees, 10 European countries, >400 campsites, >5 acco types per campsite, 4 market groups, ≈ 30 weeks open
➔ > 400 ⋅ 5 ⋅ 4 ⋅ 30 = 250 000 prices to manage each year
Problem: What do people need to pay for their stay?
6
A modern (wo)man would turn to machine learning,
so let’s think about….
A table could look like:
10
Booking
Week
Week of
your stay
Type of
house
Price Number of
bookings
Special
(holi)days
TARGET
#bookings
this week
How to model?
Easy model (like random forest).
• Iterate on feature engineering.
• Understand what’s happening.
• Setup a good training/validation
• Think about next steps
One step beyond…
Specialized model like RNN, LSTM, … which the time series components explicitly into
account.
11
Time gapPast Future
Historic information Future
Today
Historic information Future
Previous snapshot
Finding relationship Applying relationship
How to measure accuracy?
Bookings made
before “today”
Data pipeline
frequency
Max weeks into the
future we make decisions
Timeline for an example
13
Engine overview
14
Input: bookings,
availability, prices, …
Forecast engine
Price elasticity
business rules
Business rules on
min./max. prices Optimization engine
Optimal price per reservable option
market group
Demand forecast per
reservable option market
group at multiple price levels
Demand forecast at a
reference price
