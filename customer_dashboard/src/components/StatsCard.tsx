import React, { ReactNode } from 'react';
import '../styles/StatsCard.css';

interface StatsCardProps {
  title: string;
  value: string | number;
  change?: string;
  changeType?: 'positive' | 'negative' | 'neutral';
  icon?: ReactNode;
  color?: 'primary' | 'success' | 'warning' | 'error';
}

export const StatsCard: React.FunctionComponent<StatsCardProps> = ({
  title,
  value,
  change,
  changeType = 'neutral',
  icon,
  color = 'primary'
}) => {
  return (
    <div className={`stats-card stats-card--${color}`}>
      <div className="stats-card-header">
        <div className="stats-card-icon">
          {icon || (
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
              <rect x="3" y="8" width="4" height="9" rx="1" fill="currentColor"/>
              <rect x="10" y="5" width="4" height="12" rx="1" fill="currentColor"/>
              <rect x="17" y="2" width="4" height="15" rx="1" fill="currentColor"/>
            </svg>
          )}
        </div>
        <h3 className="stats-card-title">{title}</h3>
      </div>
      
      <div className="stats-card-content">
        <div className="stats-card-value">{value}</div>
        {change && (
          <div className={`stats-card-change stats-card-change--${changeType}`}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
              {changeType === 'positive' ? (
                <path d="M7 14l5-5 5 5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              ) : changeType === 'negative' ? (
                <path d="M17 10l-5 5-5-5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              ) : (
                <path d="M5 12h14" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
              )}
            </svg>
            <span>{change}</span>
          </div>
        )}
      </div>
    </div>
  );
};

export default StatsCard;
