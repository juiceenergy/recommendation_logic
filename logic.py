import requests
import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

#monthly shaping for generic residential profile.
multiple = [0.07,
	0.06,
	0.05,
	0.06,
	0.07,
	0.11,
	0.13,
	0.14,
	0.1,
	0.08,
	0.06,
	0.07]

# TDSP charges as of 7/31/2022
ONCOR = {'base':3.42, 'kwh':3.89070}
CENTERPOINT = {'base':4.39, 'kwh':3.80000}
AEP_CENTRAL = {'base':5.88, 'kwh':4.52130}
AEP_NORTH = {'base':5.88, 'kwh':4.10580}
TNMP = {'base':7.85, 'kwh':4.72740}

# TSDP codes for the Power to Choose api as of 7/31/2022
TDSP = {
    'ELSQL01DB1245281100006':ONCOR,
    'ELSQL01DB1245281100004':CENTERPOINT,
    'ELSQL01DB1245281100002':AEP_CENTRAL,
    'ELSQL01DB1245281100003':AEP_NORTH,
    'ELSQL01DB1245281100008':TNMP
    }


# Termination fee option valuation model
def binomial_lattice_option(price, low, kwhmo, months, cancel_fee, vol):
    u = np.exp(vol / 12**.5)
    d = 1/u
    p = (1-u)/(d-u)
    probs = np.array([[p, 1-p]])
    
    i = months
    prices = u ** np.arange(i) * d ** np.arange(i,0,-1) * low
    ov = np.maximum((prices - price)*kwhmo, -cancel_fee)

    for i in range(months-1,0,-1):
        prices = u ** np.arange(i) * d ** np.arange(i,0,-1) * low
        ov = np.maximum((sliding_window_view(ov, 2) * probs).sum(axis=1) + (prices - price)*kwhmo, -cancel_fee)

    return np.maximum(ov,0)



def pick_best_plan(request):
    #query powertochoose.org for available plans
    url = r"http://api.powertochoose.org/api/PowerToChoose/plans"
    params = {'zip_code': request.POST['zip_code']}
    plans = requests.get(url, params=params)
    plans = pd.DataFrame(plans.json()['data'])
    # if zip is not in texas
    # or under retail competition
    # API request will return an emtpy frame
    if plans.empty:
        return_dict = dict(
            good_to_go = False
        )
        
        return return_dict
    else:
        plans = plans.sort_values(['company_name','plan_name'])
        
        # did the subscriber request a 100% renewable plan?
        re_only = request.POST['renewable_energy_only_plan']
        re_only = re_only == 'true'
        #use a simple model to estimate power consumption
        sqft = float(request.POST.get('sq_ft'))
        estimated_annual_usage = -.000307*(sqft**2)+4*sqft + 11000
        
        # filter down to plans that are no-gimmick
        sp = plans[~plans['minimum_usage'] & ~plans['timeofuse']]
        
        # filter to plans that are at least 12 months long
        sp = sp[sp['term_value'] >= 12]
        how_many_plans = sp.shape[0]
        
        # filter to renewable plans if requested
        if re_only:
            sp = sp[sp['renewable_energy_id'] == 100]
        
        
        # calculate c/kwh charge inclusive of TDSP
        sp['metered'] = (sp['price_kwh2000']*2000 - sp['price_kwh1000']*1000)/(1000)
        # calculate monthly charge within 1 sig fig
        sp['fixed_charge'] = sp['price_kwh2000']*2000 - (sp['metered'] * 2000)

        sp['check'] = (sp['metered'] * 500 + sp['fixed_charge']) / 500
        
        # conver the cancellation fee strings into usable fields
        def parse_cancellation_fee(x):
            import re
            xx = x.split(':')
            x = xx[1]
            if 'month' in x.lower():
                per_month=True
            else:
                per_month=False
            numbers = re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", x)
            return pd.Series({'numbers':float(numbers[0]),'per_month':per_month})
        
        sp[['numbers','per_month']] = sp['pricing_details'].apply(parse_cancellation_fee)
        
        # for "$X/mo remaining" fees, assume cancellation happens in 12 months
        sp['cancellation_fee'] = (sp['per_month']*sp['numbers'] * (sp['term_value']-12) + sp['numbers']*~sp['per_month'])        
        sp['12mo cost base'] = 1/100 * (estimated_annual_usage * sp['metered'] + sp['fixed_charge'] * 12)
        sp['avg_rate'] = sp['12mo cost base'] / estimated_annual_usage
        
        # sort by two fields so that people get exactly the same recommendation when providing the same information.
        sp = sp.sort_values(list(['12mo cost base','company_name']),axis=0)
        

        sp['tte'] = (sp['term_value'] - 12) / 12 + .001
        cheapest = sp['avg_rate'].min()

        # calculate the total value of the cancellation option
        sp['cancel_option'] = sp.apply(lambda x: binomial_lattice_option(
            x['avg_rate'],
            cheapest,
            estimated_annual_usage / 12,
            x['term_value'],
            x['cancellation_fee'],
            .2
        ), axis=1)

        # convert the cancel option into c/kwh for equal comparison.
        sp['cancel_option_kwh'] = (sp['cancel_option']) / ((estimated_annual_usage / 12) * sp['term_value']) * 100
        
        # pick the best plan, factoring the fixed rate and the value of the cancel option
        sp['kwh_value'] = sp['avg_rate'] - sp['cancel_option_kwh']
        sp['kwh_value'] = sp['kwh_value'].astype(float)
        sp = sp.sort_values(list(['kwh_value','company_name']),axis=0)
        best_plans = sp.head(1)
