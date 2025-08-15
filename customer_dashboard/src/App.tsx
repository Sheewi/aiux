import * as React from 'react';
import { useState } from 'react';
import './App.css';
import DashboardHeader from './components/DashboardHeader';
import StatsCard from './components/StatsCard';
import CustomerCard from './components/CustomerCard';
import DataTable from './components/DataTable';

function App() {
  // Sample data for demonstration
  const [selectedCustomers, setSelectedCustomers] = useState<string[]>([]);
  
  const statsData = [
    {
      title: 'Total Customers',
      value: '2,543',
      change: '+12%',
      changeType: 'positive' as const,
      color: 'primary' as const,
      icon: (
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
          <path d="M16 21v-2a4 4 0 00-4-4H6a4 4 0 00-4 4v2" stroke="currentColor" strokeWidth="2"/>
          <circle cx="9" cy="7" r="4" stroke="currentColor" strokeWidth="2"/>
          <path d="M22 21v-2a4 4 0 00-3-3.87" stroke="currentColor" strokeWidth="2"/>
          <path d="M16 3.13a4 4 0 010 7.75" stroke="currentColor" strokeWidth="2"/>
        </svg>
      )
    },
    {
      title: 'Active Orders',
      value: '156',
      change: '+8%',
      changeType: 'positive' as const,
      color: 'success' as const,
      icon: (
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
          <path d="M9 11l3 3L22 4" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
          <path d="M21 12v7a2 2 0 01-2 2H5a2 2 0 01-2-2V5a2 2 0 012-2h11" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
      )
    },
    {
      title: 'Revenue',
      value: '$45,678',
      change: '+15%',
      changeType: 'positive' as const,
      color: 'warning' as const,
      icon: (
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
          <path d="M12 2v20M17 5H9.5a3.5 3.5 0 000 7h5a3.5 3.5 0 010 7H6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
      )
    },
    {
      title: 'Support Tickets',
      value: '23',
      change: '-5%',
      changeType: 'negative' as const,
      color: 'error' as const,
      icon: (
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
          <path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
      )
    }
  ];

  const sampleCustomers = [
    {
      id: '1',
      name: 'Sarah Johnson',
      email: 'sarah.johnson@example.com',
      avatar: '/api/placeholder/48/48',
      status: 'active' as const,
      lastActivity: '2 hours ago',
      value: 15420,
      location: 'New York, USA'
    },
    {
      id: '2', 
      name: 'Michael Chen',
      email: 'michael.chen@example.com',
      status: 'pending' as const,
      lastActivity: '1 day ago',
      value: 8750,
      location: 'San Francisco, USA'
    },
    {
      id: '3',
      name: 'Emma Wilson',
      email: 'emma.wilson@example.com',
      avatar: '/api/placeholder/48/48',
      status: 'inactive' as const,
      lastActivity: '1 week ago',
      value: 23100,
      location: 'London, UK'
    }
  ];

  const tableColumns = [
    {
      key: 'name',
      title: 'Customer Name',
      sortable: true,
      render: (value: string, record: any) => (
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          {record.avatar ? (
            <img src={record.avatar} alt={value} style={{ width: '2rem', height: '2rem', borderRadius: '50%' }} />
          ) : (
            <div style={{ 
              width: '2rem', 
              height: '2rem', 
              borderRadius: '50%', 
              background: '#e2e8f0', 
              display: 'flex', 
              alignItems: 'center', 
              justifyContent: 'center',
              fontSize: '0.875rem',
              fontWeight: '500'
            }}>
              {value.charAt(0)}
            </div>
          )}
          <span>{value}</span>
        </div>
      )
    },
    {
      key: 'email',
      title: 'Email',
      sortable: true
    },
    {
      key: 'status',
      title: 'Status',
      render: (value: string) => (
        <span className={`status-badge status-badge--${value}`}>
          {value}
        </span>
      )
    },
    {
      key: 'value',
      title: 'Total Value',
      sortable: true,
      render: (value: number) => (
        new Intl.NumberFormat('en-US', {
          style: 'currency',
          currency: 'USD'
        }).format(value)
      )
    },
    {
      key: 'lastActivity',
      title: 'Last Activity',
      sortable: true
    }
  ];

  const handleSearch = (query: string) => {
    console.log('Search:', query);
  };

  const handleCustomerAction = (action: string, customerId: string) => {
    console.log(`${action} customer:`, customerId);
  };

  return (
    <div className="App">
      <DashboardHeader
        title="Customer Dashboard"
        userName="Admin User"
        notifications={5}
        onSearch={handleSearch}
        onProfileClick={() => console.log('Profile clicked')}
        onNotificationClick={() => console.log('Notifications clicked')}
      />
      
      <main className="dashboard-main">
        {/* Stats Overview */}
        <section className="stats-section">
          <div className="stats-grid">
            {statsData.map((stat, index) => (
              <StatsCard
                key={index}
                title={stat.title}
                value={stat.value}
                change={stat.change}
                changeType={stat.changeType}
                color={stat.color}
                icon={stat.icon}
              />
            ))}
          </div>
        </section>

        {/* Customer Cards Grid */}
        <section className="customers-section">
          <h2 className="section-title">Recent Customers</h2>
          <div className="customers-grid">
            {sampleCustomers.map((customer) => (
              <CustomerCard
                key={customer.id}
                customer={customer}
                onViewDetails={(id) => handleCustomerAction('view', id)}
                onEdit={(id) => handleCustomerAction('edit', id)}
                onDelete={(id) => handleCustomerAction('delete', id)}
              />
            ))}
          </div>
        </section>

        {/* Data Table */}
        <section className="table-section">
          <h2 className="section-title">All Customers</h2>
          <DataTable
            columns={tableColumns}
            data={sampleCustomers}
            pagination={{
              current: 1,
              pageSize: 10,
              total: sampleCustomers.length,
              onChange: (page) => console.log('Page changed:', page)
            }}
            rowSelection={{
              selectedRowKeys: selectedCustomers,
              onChange: setSelectedCustomers
            }}
            onRowClick={(record) => console.log('Row clicked:', record)}
          />
        </section>
      </main>
    </div>
  );
}

export default App;
