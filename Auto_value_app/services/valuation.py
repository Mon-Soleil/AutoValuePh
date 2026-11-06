"""Turns a raw ML prediction into the breakdown shown on the result card."""

# An asking price this many percent below / above the ML value changes the verdict.
GOOD_DEAL_PCT = 10


def build_valuation(vehicles, vehicle, predicted, price_range, region=None, asking_price=None):
    """Assemble the result card data.

    The deal verdict compares ONLY the price the user typed in (``asking_price``)
    to the model's value. With no price there is nothing to judge, so the verdict
    is "Estimate Only". Comparable bank listings are shown as supporting context.
    """
    comps = vehicles.comparables(
        vehicle['make'], vehicle['model'], vehicle['year'], region=region,
    )

    reference = (
        {'label': 'Your asking / bid price', 'price': asking_price, 'kind': 'asking'}
        if asking_price else None
    )

    return {
        'predicted_price': predicted,
        'formatted_price': f'₱{predicted:,.0f}',
        'range': (
            {'low': price_range[0], 'high': price_range[1]} if price_range else None
        ),
        'reference': reference,
        'verdict': _verdict(predicted, reference),
        'comparables': comps,
    }


def _verdict(predicted, reference):
    if not reference or predicted <= 0:
        return {'label': 'Estimate Only', 'tone': 'neutral', 'diff_pct': None}

    # Positive => the asking price is BELOW the model's market value.
    diff_pct = round((predicted - reference['price']) / predicted * 100)
    if diff_pct >= GOOD_DEAL_PCT:
        label, tone = 'Great Deal', 'success'
    elif diff_pct > -GOOD_DEAL_PCT:
        label, tone = 'Fair Price', 'warning'
    else:
        label, tone = 'Above Market', 'danger'
    return {'label': label, 'tone': tone, 'diff_pct': diff_pct}