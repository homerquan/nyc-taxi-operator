import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { SummaryCard } from './SummaryCard';

describe('SummaryCard', () => {
  it('renders values supplied by the forecast rather than placeholder data', () => {
    render(<SummaryCard label="PREDICTED TRIPS" value="1,234" detail="Next hour" icon={null}/>);
    expect(screen.getByText('1,234')).toBeTruthy();
    expect(screen.getByText('Next hour')).toBeTruthy();
  });
});
