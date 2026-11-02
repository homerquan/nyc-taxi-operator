import { useEffect, useMemo, useState } from 'react';
import DeckGL from '@deck.gl/react';
import { ArcLayer, GeoJsonLayer, ScatterplotLayer } from '@deck.gl/layers';
import type { PickingInfo } from '@deck.gl/core';
import Map from 'react-map-gl/maplibre';
import 'maplibre-gl/dist/maplibre-gl.css';
import type { Flow, ZoneFeature, ZoneProperties, Zones } from './types';
import { number } from './api';

type Props = { zones: Zones; flows: Flow[]; selectedZone: number | null; heat: Record<string, number> | null; onZone: (zone: number) => void; onFlow: (flow: Flow) => void };

export function MapView({ zones, flows, selectedZone, heat, onZone, onFlow }: Props) {
  const [phase, setPhase] = useState(0);
  useEffect(() => { const timer = window.setInterval(() => setPhase(x => (x + 0.025) % 1), 80); return () => window.clearInterval(timer); }, []);
  const maxHeat = Math.max(1, ...Object.values(heat || {}));
  const maxFlow = Math.max(0.001, ...flows.map(f => f.value));
  const layers = useMemo(() => [
    new GeoJsonLayer({ id: 'zones', data: zones as never, pickable: true, stroked: true, filled: true,
      getLineColor: [125, 145, 143, 110], getLineWidth: 1, lineWidthMinPixels: 0.6,
      getFillColor: feature => {
        const properties = feature.properties as ZoneProperties;
        if (properties.zone_id === selectedZone) return [25, 109, 101, 100];
        const value = heat?.[String(properties.zone_id)] || 0;
        return value ? [26, 130, 115, Math.round(25 + 145 * Math.sqrt(value / maxHeat))] : [255, 255, 255, 10];
      },
      updateTriggers: { getFillColor: [selectedZone, heat, maxHeat] },
      onClick: (info: PickingInfo<ZoneFeature>) => { if (info.object) onZone(info.object.properties.zone_id); }
    }),
    new ScatterplotLayer<ZoneFeature>({ id: 'centroids', data: zones.features, pickable: false, getPosition: f => [f.properties.longitude, f.properties.latitude], getRadius: 2, radiusMinPixels: 1.2, getFillColor: [38, 74, 70, 115] }),
    new ArcLayer<Flow>({ id: 'flows', data: flows, pickable: true, getSourcePosition: d => d.from, getTargetPosition: d => d.to,
      getSourceColor: d => flowColor(d, maxFlow, 155),
      getTargetColor: d => flowColor(d, maxFlow, 225),
      getWidth: d => Math.max(1.4, Math.min(8, 1 + Math.sqrt(d.value)*1.3)), widthUnits: 'pixels', getHeight: 0.3,
      onClick: (info: PickingInfo<Flow>) => { if (info.object) onFlow(info.object); }
    }),
    new ScatterplotLayer<Flow>({ id: 'within-zone', data: flows.filter(f => f.source === f.target), pickable: true,
      getPosition: d => d.from, getRadius: d => Math.max(30, Math.min(180, 30 + Math.sqrt(d.value)*25)),
      radiusUnits: 'meters', stroked: true, filled: false, lineWidthMinPixels: 2, getLineColor: d => flowColor(d, maxFlow, 220),
      onClick: (info: PickingInfo<Flow>) => { if (info.object) onFlow(info.object); } }),
    new ScatterplotLayer<Flow>({ id: 'moving-flow', data: flows.filter(f => f.source !== f.target), pickable: false,
      getPosition: d => { const t = (phase + ((d.source*19+d.target*7)%31)/31)%1; return [d.from[0]+(d.to[0]-d.from[0])*t, d.from[1]+(d.to[1]-d.from[1])*t+0.008*Math.sin(Math.PI*t)]; },
      getRadius: 2, radiusMinPixels: 2, getFillColor: [245, 184, 82, 220], updateTriggers: { getPosition: [phase] } })
  ], [zones, flows, selectedZone, heat, maxHeat, maxFlow, phase, onZone, onFlow]);

  return <DeckGL initialViewState={{ longitude: -73.96, latitude: 40.73, zoom: 10.25, pitch: 0, bearing: 0 }} controller layers={layers}
    getTooltip={({ object }: PickingInfo<Flow | ZoneFeature>) => object ? ('from' in object
      ? `${object.source} → ${object.target}\n${number(object.value, 1)} trips` : `${object.properties.zone}\n${object.properties.borough}`) : null}>
    <Map mapStyle="https://tiles.openfreemap.org/styles/positron" attributionControl={{}} />
  </DeckGL>;
}

function flowColor(flow: Flow, maxFlow: number, alpha: number): [number,number,number,number] {
  const t = Math.sqrt(Math.max(0, flow.value) / maxFlow);
  if (flow.error != null && flow.error < 0) return [220, Math.round(164-50*t), Math.round(112-45*t), alpha];
  return [Math.round(219-198*t), Math.round(190-82*t), Math.round(122-19*t), alpha];
}
