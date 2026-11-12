/* AutoValue PH – frontend
 *
 * All vehicle data comes from the Flask backend (backed by dataset.json):
 *   GET  /api/v1/options  -> next cascading dropdown's values
 *   POST /api/v1/predict  -> live valuation from the trained ML pipeline
 */
document.addEventListener('DOMContentLoaded', () => {
  const $ = (selector) => document.querySelector(selector);
  const $$ = (selector) => [...document.querySelectorAll(selector)];

  const peso = new Intl.NumberFormat('en-PH', {
    style: 'currency', currency: 'PHP', maximumFractionDigits: 0,
  });
  const fmtPeso = (n) => peso.format(n);
  const fmtNum = (n) => Number(n).toLocaleString('en-PH');

  // ------------------------------------------------------------------ API --

  async function api(url, options) {
    let response;
    try {
      response = await fetch(url, options);
    } catch {
      throw new Error('Could not reach the server. Is the Flask app running?');
    }
    const body = await response.json().catch(() => ({}));
    if (!response.ok || body.status === 'error') {
      throw new Error(body.message || `Request failed (${response.status}).`);
    }
    return body;
  }

  const apiGet = (url, params = {}) => api(`${url}?${new URLSearchParams(params)}`);
  const apiPost = (url, payload) => api(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });

  // --------------------------------------------------------------- Form --

  const form = $('#valuation-form');
  const errorBox = $('#form-error');

  const showError = (message) => {
    errorBox.textContent = message;
    errorBox.hidden = !message;
  };

  // ---------------------------------------------------- Cascading dropdowns --

  const cascade = $$('#fields-ymm select[data-level]'); // year -> make -> ... -> fuel_type
  let cascadeRun = 0; // lets us ignore responses from an outdated selection

  function resetFrom(index) {
    cascade.slice(index).forEach((select) => {
      select.replaceChildren(new Option('-- Select --', ''));
      select.disabled = true;
    });
  }

  function fillSelect(select, values) {
    select.replaceChildren(
      new Option(select.dataset.placeholder || '-- Select --', ''),
      ...values.map((value) => new Option(String(value), String(value))),
    );
    select.disabled = values.length === 0;
  }

  async function loadNext(index, run) {
    const next = cascade[index + 1];
    if (!next || !cascade[index].value) return;

    const params = { level: next.dataset.level };
    cascade.slice(0, index + 1).forEach((s) => { params[s.dataset.level] = s.value; });

    const { options } = await apiGet('/api/v1/options', params);
    if (run !== cascadeRun) return; // user changed something meanwhile

    fillSelect(next, options);
    if (options.length === 1) { // only one possibility: pick it and continue down the chain
      next.value = String(options[0]);
      resetFrom(index + 2);
      await loadNext(index + 1, run);
    }
  }

  cascade.forEach((select, index) => {
    select.addEventListener('change', async () => {
      const run = ++cascadeRun;
      resetFrom(index + 1);
      try {
        await loadNext(index, run);
      } catch (err) {
        showError(err.message);
      }
    });
  });

  // ------------------------------------------------------------ Rendering --

  const setText = (id, value) => {
    const el = $(id);
    if (el) el.textContent = value; // skip tiles that aren't in the page instead of crashing
  };
  const reportTitle = $('#modal-subtitle').textContent.trim(); // "<App name> Verified Report"

  // The dataset uses the literal text "N/A" when a unit has no trim; don't show it as a trim name.
  const trimLabel = (v) => (!v || /^#?n\/?a$/i.test(v.trim()) ? '' : v);

  function renderSpecs(unit, regionLabel) {
    const title = [unit.year, unit.make, unit.model, trimLabel(unit.variant)].filter(Boolean).join(' ');
    setText('#car-title', title);
    setText('#spec-model', `${unit.make} ${unit.model}`);
    setText('#spec-trim', trimLabel(unit.variant) || 'Base Trim');
    setText('#spec-transmission', unit.transmission);
    setText('#spec-body-type', unit.body_type);
    setText('#spec-engine', `${unit.engine_displacement} L`);
    setText('#spec-fuel', unit.fuel_type);
    setText('#spec-odo', `${fmtNum(unit.mileage)} km`);
    setText('#spec-year', unit.year);
    setText('#spec-region', regionLabel || 'All regions');
    setText('#modal-subtitle', `${reportTitle} • ${unit.year} ${unit.make} ${unit.model}`);
  }

  // Build "text <strong>bold</strong> text" safely (no innerHTML with data).
  function renderRich(element, parts) {
    element.replaceChildren(...parts.map((part) => {
      if (typeof part === 'string') return document.createTextNode(part);
      const strong = document.createElement('strong');
      strong.textContent = part.bold;
      return strong;
    }));
  }

  function verdictSentence(result) {
    const { reference, verdict } = result;
    if (!reference) {
      return ['Estimated market value is ', { bold: result.formatted_price },
        '. Enter an asking / bid price to see whether it is a good deal.'];
    }
    const pct = Math.abs(verdict.diff_pct);
    const relation = pct < 1 ? ['in line with the ML market value'] : [
      { bold: `${pct}% ${verdict.diff_pct > 0 ? 'below' : 'above'}` }, ' the ML market value'];
    return ['Your asking price of ', { bold: fmtPeso(reference.price) }, ' is approximately ',
      ...relation, ` of ${fmtPeso(result.predicted_price)}.`];
  }

  function breakdownRows(result) {
    const rows = [];
    const { comparables: comps, reference } = result;
    if (reference) rows.push([reference.label, fmtPeso(reference.price)]);
    if (comps.count) {
      rows.push([`Median bank listing (${comps.scope})`, fmtPeso(comps.median)]);
      rows.push([
        `Listings found • price range`,
        `${comps.count} • ${fmtPeso(comps.min)} – ${fmtPeso(comps.max)}`,
      ]);
    }
    if (comps.region) {
      rows.push([
        `Median in ${comps.region.label}`,
        comps.region.count
          ? `${fmtPeso(comps.region.median)} (${comps.region.count} listings)`
          : 'No comparable listings',
      ]);
    }
    return rows;
  }

  function renderResult(result) {
    const { verdict } = result;

    const badge = $('#verdict-badge');
    badge.textContent = verdict.label;
    badge.className = `badge-deal ${verdict.tone === 'success' ? '' : verdict.tone}`.trim();

    renderRich($('#verdict-text'), verdictSentence(result));

    setText('#value-big', result.formatted_price);
    setText('#value-range', result.range
      ? `Likely range ${fmtPeso(result.range.low)} – ${fmtPeso(result.range.high)}`
      : '');
    $('#value-hero').hidden = false;

    const list = $('#breakdown-list');
    list.replaceChildren(...breakdownRows(result).map(([label, value]) => {
      const row = document.createElement('div');
      row.className = 'breakdown-row';
      const dt = document.createElement('dt');
      const dd = document.createElement('dd');
      dt.textContent = label;
      dd.textContent = value;
      row.append(dt, dd);
      return row;
    }));
    list.hidden = list.children.length === 0;

    // brief highlight, as in the original design
    const box = $('#verdict-box');
    box.style.borderColor = 'var(--success-green)';
    setTimeout(() => { box.style.borderColor = 'var(--border-color)'; }, 1000);
  }

  // --------------------------------------------------------------- Submit --

  const regionSelect = $('#region-select');
  const submitBtn = $('#btn-submit');

  function unitFromDropdowns() {
    const value = (level) => $(`#fields-ymm [data-level="${level}"]`).value.trim();
    const unit = {
      year: Number(value('year')),
      make: value('make'),
      model: value('model'),
      variant: value('variant'),
      body_type: value('body_type'),
      engine_displacement: Number(value('engine_displacement')),
      transmission: value('transmission'),
      fuel_type: value('fuel_type'),
      mileage: Number($('#mileage-input').value),
      asking_price: $('#asking-price-input').value ? Number($('#asking-price-input').value) : null,
    };
    return unit;
  }

  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    showError('');
    submitBtn.disabled = true;
    submitBtn.textContent = 'Valuing…';
    $('#verdict-box').classList.add('is-loading');

    try {
      const unit = unitFromDropdowns();
      const regionKey = regionSelect.value || unit.region || '';
      const regionLabel = regionSelect.value
        ? regionSelect.selectedOptions[0].textContent
        : (unit.region_label || 'All regions');

      renderSpecs(unit, regionLabel);

      const result = await apiPost('/api/v1/predict', {
        year: unit.year,
        mileage: unit.mileage,
        engine_displacement: unit.engine_displacement,
        make: unit.make,
        body_type: unit.body_type,
        model: unit.model,
        variant: unit.variant,
        transmission: unit.transmission,
        fuel_type: unit.fuel_type,
        region: regionKey,
        asking_price: unit.asking_price,
      });
      renderResult(result);
    } catch (err) {
      showError(err.message);
    } finally {
      submitBtn.disabled = false;
      submitBtn.textContent = 'See Vehicle Value ↗';
      $('#verdict-box').classList.remove('is-loading');
    }
  });

  // ------------------------------------------------------------ Report modal --

  const modal = $('#report-modal');
  $('#btn-report').addEventListener('click', () => modal.classList.add('active'));
  $('#btn-close-modal').addEventListener('click', () => modal.classList.remove('active'));
  $('#btn-download-sim').addEventListener('click', () => window.print());
});