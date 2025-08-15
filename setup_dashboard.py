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
sys.path.append('/media/r/Workspace/tools')

try:
    # Try different import approaches - try direct import first since we added the design path
        try:
            # Try as a direct import
            from figma_tool import figma_tool
        except ImportError:
            try:
                # Try with tools as the root package
                from tools.design.figma_integration import figma_tool
        except ImportError:
            # Try with design as the package
            from design.figma_integration import figma_tool
except ImportError:
    # Create mock implementation if module isn't found
    print("Warning: figma_integration module not found, using mock implementation")
    class FigmaTool:
        async def initialize(self):
            print("Mock: Figma initialized")
            return True
            
        async def create_design_system(self, name, colors):
            return {'success': True, 'design_system_id': 'mock-ds-123'}
            
        async def create_component(self, name, description):
            return {'success': True}
            
        async def get_design_handoff(self, file_id):
            return {'success': True, 'components_count': 10, 'tokens_count': 25}
            
        def get_status(self):
            return {
                'connected': False, 
                'components_count': 10,
                'design_tokens_count': 25,
                'projects_count': 1
            }
    
    figma_tool = FigmaTool()

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
    
    # Create Flutter app structure
    print("\n5. Setting up Flutter application...")
    await setup_flutter_structure()
    
    # Generate final summary
    print("\n" + "=" * 60)
    print("🎉 Dashboard Setup Complete!")
    print("=" * 60)
    
    status = figma_tool.get_status()
    print(f"Figma Status: {'Connected' if status['connected'] else 'Simulation Mode'}")
    print(f"Components: {status['components_count']}")
    print(f"Design Tokens: {status['design_tokens_count']}")
    print(f"Projects: {status['projects_count']}")
    
    print("\n📁 Generated Files:")
    print("   /media/r/Workspace/design_systems/")
    print("   /media/r/Workspace/flutter_app/lib/")
    print("   /media/r/Workspace/design_handoff/")
    print("   /media/r/Workspace/flutter_app/")
    
    print("\n🚀 Next Steps:")
    print("   1. Add your Figma API token for real integration")
    print("   2. Run './setup_flutter_server.sh' to install dependencies")
    print("   3. Run 'flutter run' in flutter_app directory")
    print("   4. Customize Flutter widgets based on your requirements")
    
    return True

async def setup_flutter_structure():
    """Setup Flutter application structure"""
    base_path = "/media/r/Workspace/flutter_app/lib"
    # Create necessary directories for Flutter app
    directories = [
        "components",
        "screens",
        "models",
        "utils",
        "widgets"
    ]
    for directory in directories:
        os.makedirs(os.path.join(base_path, directory), exist_ok=True)
    # Create a sample main.dart if not exists
    main_dart = os.path.join(base_path, "main.dart")
    if not os.path.exists(main_dart):
        with open(main_dart, "w") as f:
            f.write('''import 'package:flutter/material.dart';\n\nvoid main() {\n  runApp(const MyApp());\n}\n\nclass MyApp extends StatelessWidget {\n  const MyApp({super.key});\n\n  @override\n  Widget build(BuildContext context) {\n    return MaterialApp(\n      title: 'Customer Dashboard',\n      theme: ThemeData(\n        primarySwatch: Colors.blue,\n      ),\n      home: const DashboardScreen(),\n    );\n  }\n}\n\nclass DashboardScreen extends StatelessWidget {\n  const DashboardScreen({super.key});\n\n  @override\n  Widget build(BuildContext context) {\n    return Scaffold(\n      appBar: AppBar(title: const Text('Customer Dashboard')),\n      body: const Center(child: Text('Welcome to the Customer Dashboard!')),\n    );\n  }\n}\n''')
    print("✅ Flutter structure created")

if __name__ == "__main__":
    asyncio.run(setup_dashboard())
