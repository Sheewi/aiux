#!/usr/bin/env python3
"""
Dashboard Setup with Figma Integration
Generates a complete customer dashboard using Figma design system
"""

import asyncio
import os
import sys
import json
from pathlib import Path

# Add tools to path
sys.path.append('/media/r/Workspace/tools/design')
from figma_integration import figma_tool

async def setup_dashboard():
    """Setup complete dashboard with Figma integration"""
    print("🎨 Setting up Customer Dashboard with Figma Integration")
    print("=" * 60)
    
    # Initialize Figma
    print("1. Initializing Figma integration...")
    success = await figma_tool.initialize()
    if not success:
        print("❌ Figma initialization failed")
        return False
    print("✅ Figma integration ready")
    
    # Create design system
    print("\\n2. Creating design system...")
    brand_colors = {
        'primary': '#2563EB',
        'secondary': '#64748B', 
        'success': '#10B981',
        'warning': '#F59E0B',
        'error': '#EF4444',
        'info': '#3B82F6',
        'neutral': '#6B7280',
        'accent': '#8B5CF6'
    }
    
    ds_result = await figma_tool.create_design_system(
        "Customer Dashboard Design System",
        brand_colors
    )
    
    if ds_result['success']:
        print(f"✅ Design system created: {ds_result['design_system_id']}")
        print(f"   Colors: {len(brand_colors)}")
    else:
        print("❌ Design system creation failed")
        return False
    
    # Create dashboard components
    print("\\n3. Creating dashboard components...")
    
    components_to_create = [
        {
            'name': 'Dashboard Header',
            'description': 'Navigation header with search, notifications, and user menu',
            'type': 'header'
        },
        {
            'name': 'Stats Card',
            'description': 'Metric display card with value, change indicator, and icon',
            'type': 'card'
        },
        {
            'name': 'Customer Card',
            'description': 'Individual customer information card with actions',
            'type': 'card'
        },
        {
            'name': 'Data Table',
            'description': 'Sortable, paginated table with row selection',
            'type': 'table'
        },
        {
            'name': 'Button',
            'description': 'Primary action button with variants',
            'type': 'button'
        },
        {
            'name': 'Input Field',
            'description': 'Form input with validation states',
            'type': 'input'
        },
        {
            'name': 'Modal',
            'description': 'Overlay dialog for user interactions',
            'type': 'modal'
        },
        {
            'name': 'Sidebar',
            'description': 'Navigation sidebar with menu items',
            'type': 'navigation'
        },
        {
            'name': 'Badge',
            'description': 'Status indicator and labels',
            'type': 'indicator'
        },
        {
            'name': 'Loading Spinner',
            'description': 'Loading state indicator',
            'type': 'feedback'
        }
    ]
    
    created_components = []
    for comp_config in components_to_create:
        comp_result = await figma_tool.create_component(
            comp_config['name'],
            comp_config['description']
        )
        
        if comp_result['success']:
            created_components.append(comp_config['name'])
            print(f"   ✅ {comp_config['name']}")
        else:
            print(f"   ❌ {comp_config['name']} - {comp_result.get('error', 'Unknown error')}")
    
    print(f"\\n✅ Created {len(created_components)} components")
    
    # Generate assets and handoff
    print("\\n4. Generating design handoff...")
    handoff_result = await figma_tool.get_design_handoff('dashboard_file')
    
    if handoff_result['success']:
        print(f"✅ Design handoff generated")
        print(f"   Components: {handoff_result['components_count']}")
        print(f"   Design tokens: {handoff_result['tokens_count']}")
    else:
        print("❌ Design handoff generation failed")
    
    # Create React app structure
    print("\\n5. Setting up React application...")
    await setup_react_structure()
    
    # Generate final summary
    print("\\n" + "=" * 60)
    print("🎉 Dashboard Setup Complete!")
    print("=" * 60)
    
    status = figma_tool.get_status()
    print(f"Figma Status: {'Connected' if status['connected'] else 'Simulation Mode'}")
    print(f"Components: {status['components_count']}")
    print(f"Design Tokens: {status['design_tokens_count']}")
    print(f"Projects: {status['projects_count']}")
    
    print("\\n📁 Generated Files:")
    print("   /media/r/Workspace/design_systems/")
    print("   /media/r/Workspace/ui/components/")
    print("   /media/r/Workspace/design_handoff/")
    print("   /media/r/Workspace/customer_dashboard/")
    
    print("\\n🚀 Next Steps:")
    print("   1. Add your Figma API token for real integration")
    print("   2. Run 'npm start' in customer_dashboard directory")
    print("   3. Customize components based on your requirements")
    
    return True

async def setup_react_structure():
    """Setup React application structure"""
    base_path = "/media/r/Workspace/customer_dashboard"
    
    # Create necessary directories
    directories = [
        "src/components",
        "src/hooks",
        "src/utils",
        "src/styles",
        "src/types",
        "public"
    ]
    
    for directory in directories:
        os.makedirs(os.path.join(base_path, directory), exist_ok=True)
    
    # Create types file
    types_content = '''export interface Customer {
  id: string;
  name: string;
  email: string;
  avatar?: string;
  status: 'active' | 'inactive' | 'pending';
  lastActivity: string;
  value: number;
  location?: string;
}

export interface StatsData {
  title: string;
  value: string | number;
  change?: string;
  changeType?: 'positive' | 'negative' | 'neutral';
  color?: 'primary' | 'success' | 'warning' | 'error';
  icon?: React.ReactNode;
}

export interface TableColumn {
  key: string;
  title: string;
  width?: string;
  sortable?: boolean;
  render?: (value: any, record: any) => React.ReactNode;
}'''
    
    with open(os.path.join(base_path, 'src/types/index.ts'), 'w') as f:
        f.write(types_content)
    
    # Create utils file
    utils_content = '''export const formatCurrency = (amount: number): string => {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD'
  }).format(amount);
};

export const formatDate = (date: string | Date): string => {
  return new Intl.DateTimeFormat('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric'
  }).format(new Date(date));
};

export const getStatusColor = (status: string): string => {
  const colors = {
    active: 'var(--color-success)',
    inactive: 'var(--color-neutral)',
    pending: 'var(--color-warning)',
    error: 'var(--color-error)'
  };
  return colors[status as keyof typeof colors] || colors.neutral;
};'''
    
    with open(os.path.join(base_path, 'src/utils/index.ts'), 'w') as f:
        f.write(utils_content)
    
    # Create hooks file
    hooks_content = '''import { useState, useMemo } from 'react';

export const useSearch = <T>(data: T[], searchFields: (keyof T)[]) => {
  const [searchQuery, setSearchQuery] = useState('');
  
  const filteredData = useMemo(() => {
    if (!searchQuery) return data;
    
    return data.filter(item =>
      searchFields.some(field =>
        String(item[field]).toLowerCase().includes(searchQuery.toLowerCase())
      )
    );
  }, [data, searchQuery, searchFields]);
  
  return { searchQuery, setSearchQuery, filteredData };
};

export const useSort = <T>(data: T[]) => {
  const [sortConfig, setSortConfig] = useState<{
    key: keyof T;
    direction: 'asc' | 'desc';
  } | null>(null);
  
  const sortedData = useMemo(() => {
    if (!sortConfig) return data;
    
    return [...data].sort((a, b) => {
      const aValue = a[sortConfig.key];
      const bValue = b[sortConfig.key];
      
      if (aValue < bValue) {
        return sortConfig.direction === 'asc' ? -1 : 1;
      }
      if (aValue > bValue) {
        return sortConfig.direction === 'asc' ? 1 : -1;
      }
      return 0;
    });
  }, [data, sortConfig]);
  
  return { sortedData, sortConfig, setSortConfig };
};'''
    
    with open(os.path.join(base_path, 'src/hooks/index.ts'), 'w') as f:
        f.write(hooks_content)
    
    print("✅ React structure created")

if __name__ == "__main__":
    asyncio.run(setup_dashboard())
