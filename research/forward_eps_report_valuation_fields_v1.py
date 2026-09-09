"""只提取研报基本资料中明确的报价和合理估值，不推算精确股本。"""
from __future__ import annotations

from decimal import Decimal
import re
from research.financial_annual_components_v1 import norm,decimal

NUMBER=r'(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?'


def scalar(matches,label):
    if len(matches)!=1:
        return {'status':'NO_VIEW_MISSING_OR_NONUNIQUE_'+label,'value':None,'raw':None}
    match=matches[0]
    value=decimal(match.group('value'))
    if value<=0:
        return {'status':'NO_VIEW_NONPOSITIVE_'+label,'value':None,'raw':match.group(0)}
    return {'status':'EXPLICIT_'+label,'value':str(value),'raw':match.group(0)}


def parse_front_page(original):
    text=norm(original)
    quote_matches=list(re.finditer(r'(?m)^[ \t]*收盘价[ \t]+(?P<value>'+NUMBER+r')[ \t]*元[ \t]*$',text))
    quote=scalar(quote_matches,'CLOSE_PRICE_YUAN_LABEL')
    targets=list(re.finditer(r'(?m)^[ \t]*合理估值[ \t]+(?P<lower>'+NUMBER+r')(?:[ \t]*[-—~～至][ \t]*(?P<upper>'+NUMBER+r'))?[ \t]*元[ \t]*$',text))
    target={'status':'NO_VIEW_TARGET_MISSING_OR_NONUNIQUE','lower':None,'upper':None,'raw':None}
    if len(targets)==1:
        match=targets[0];low=decimal(match.group('lower'));high=decimal(match.group('upper') or match.group('lower'))
        target={'status':'EXPLICIT_TARGET_INTERVAL' if 0<low<=high else 'NO_VIEW_INVALID_TARGET_INTERVAL',
                'lower':str(low),'upper':str(high),'raw':match.group(0)}
    caps=list(re.finditer(r'(?m)^[ \t]*总市值/流通市值[ \t]+(?P<value>'+NUMBER+r')/(?P<float>'+NUMBER+r')[ \t]*百万元[ \t]*$',text))
    cap=scalar(caps,'TOTAL_MARKET_CAP_MILLION_LABEL')
    result={'quote':quote,'target':target,'market_cap':cap,'target_upside':None,
            'target_is_current_month_end_fair_value':False,'exact_total_shares_inferred':False,
            'valuation_method_inferred_from_target':False}
    if quote['value'] is not None and target['status']=='EXPLICIT_TARGET_INTERVAL':
        q=Decimal(quote['value']);low=Decimal(target['lower']);high=Decimal(target['upper'])
        result['target_upside']={'lower':str(low/q-1),'midpoint':str((low+high)/2/q-1),'upper':str(high/q-1),
                                 'clock':'REPORT_OWN_REFERENCE_PRICE_NOT_MONTH_END_MARK'}
    return result
