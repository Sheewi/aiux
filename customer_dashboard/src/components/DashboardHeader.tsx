import React from 'react';
import '../styles/DashboardHeader.css';

interface DashboardHeaderProps {
  title?: string;
  userAvatar?: string;
  userName?: string;
  notifications?: number;
  onSearch?: (query: string) => void;
  onProfileClick?: () => void;
  onNotificationClick?: () => void;
}

export const DashboardHeader = ({
  title = 'Customer Dashboard',
  userAvatar = '/api/placeholder/32/32',
  userName = 'John Doe',
  notifications = 0,
  onSearch,
  onProfileClick,
  onNotificationClick
}: DashboardHeaderProps) => {
  return (
    <header className="dashboard-header">
      <div className="header-left">
        <h1 className="dashboard-title">{title}</h1>
      </div>
      
      <div className="header-center">
        <div className="search-container">
          <input
            type="text"
            placeholder="Search customers..."
            className="search-input"
            onChange={(e) => onSearch?.(e.target.value)}
          />
          <svg className="search-icon" width="20" height="20" viewBox="0 0 24 24" fill="none">
            <path d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        </div>
      </div>
      
      <div className="header-right">
        <button 
          className="notification-btn"
          onClick={onNotificationClick}
        >
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
            <path d="M18 8A6 6 0 006 8c0 7-3 9-3 9h18s-3-2-3-9" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            <path d="M13.73 21a2 2 0 01-3.46 0" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
          {notifications > 0 && (
            <span className="notification-badge">{notifications}</span>
          )}
        </button>
        
        <div className="user-menu" onClick={onProfileClick}>
          <img src={userAvatar} alt={userName} className="user-avatar" />
          <span className="user-name">{userName}</span>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
            <path d="M6 9l6 6 6-6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        </div>
      </div>
    </header>
  );
};

export default DashboardHeader;
