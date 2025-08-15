import 'package:flutter/material.dart';

void main() {
  runApp(const GrokApp());
}

class GrokApp extends StatelessWidget {
  const GrokApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Grok AI Assistant',
      theme: ThemeData.dark().copyWith(
        scaffoldBackgroundColor: const Color(0xFF000000),
        appBarTheme: const AppBarTheme(
          backgroundColor: Color(0xFF000000),
          elevation: 0,
          foregroundColor: Color(0xFFFFFFFF),
        ),
      ),
      home: const GrokHomePage(),
      debugShowCheckedModeBanner: false,
    );
  }
}

class GrokHomePage extends StatefulWidget {
  const GrokHomePage({super.key});

  @override
  State<GrokHomePage> createState() => _GrokHomePageState();
}

class _GrokHomePageState extends State<GrokHomePage> {
  final TextEditingController _messageController = TextEditingController();
  final List<ChatMessage> _messages = [
    ChatMessage(
      text: "Hello! I'm your AI assistant. How can I help you today?",
      isUser: false,
      timestamp: DateTime.now().subtract(const Duration(minutes: 5)),
    ),
  ];

  void _sendMessage() {
    if (_messageController.text.trim().isEmpty) return;

    setState(() {
      _messages.add(ChatMessage(
        text: _messageController.text,
        isUser: true,
        timestamp: DateTime.now(),
      ));
    });

    // Simulate AI response
    Future.delayed(const Duration(milliseconds: 1000), () {
      setState(() {
        _messages.add(ChatMessage(
          text: "I understand your query. Let me help you with that!",
          isUser: false,
          timestamp: DateTime.now(),
        ));
      });
    });

    _messageController.clear();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF000000),
      body: Row(
        children: [
          // Sidebar
          Container(
            width: 300,
            color: const Color(0xFF000000),
            child: Column(
              children: [
                // Header
                Container(
                  padding: const EdgeInsets.all(24),
                  child: const Row(
                    children: [
                      Icon(
                        Icons.psychology,
                        color: Color(0xFF00D4FF),
                        size: 32,
                      ),
                      SizedBox(width: 16),
                      Text(
                        'AI Assistant',
                        style: TextStyle(
                          fontSize: 24,
                          fontWeight: FontWeight.bold,
                          color: Color(0xFFFFFFFF),
                        ),
                      ),
                    ],
                  ),
                ),
                
                // New Chat Button
                Container(
                  margin: const EdgeInsets.symmetric(horizontal: 16),
                  width: double.infinity,
                  child: ElevatedButton(
                    onPressed: () {
                      setState(() {
                        _messages.clear();
                        _messages.add(ChatMessage(
                          text: "Hello! I'm your AI assistant. How can I help you today?",
                          isUser: false,
                          timestamp: DateTime.now(),
                        ));
                      });
                    },
                    style: ElevatedButton.styleFrom(
                      backgroundColor: const Color(0xFF00D4FF),
                      foregroundColor: const Color(0xFF000000),
                      padding: const EdgeInsets.symmetric(vertical: 16),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(8),
                      ),
                    ),
                    child: const Text(
                      'New Chat',
                      style: TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                  ),
                ),
                
                const SizedBox(height: 32),
                
                // Chat History
                Expanded(
                  child: ListView(
                    padding: const EdgeInsets.symmetric(horizontal: 16),
                    children: [
                      _buildSidebarItem('Previous Chat 1'),
                      _buildSidebarItem('Previous Chat 2'),
                      _buildSidebarItem('Previous Chat 3'),
                      const SizedBox(height: 32),
                      _buildSidebarItem('Settings'),
                      _buildSidebarItem('Help'),
                    ],
                  ),
                ),
              ],
            ),
          ),
          
          // Vertical divider
          Container(
            width: 1,
            color: const Color(0xFF333333),
          ),
          
          // Main Chat Area
          Expanded(
            child: Container(
              color: const Color(0xFF000000),
              child: Column(
                children: [
                  // Chat Header
                  Container(
                    padding: const EdgeInsets.all(24),
                    child: const Row(
                      children: [
                        Text(
                          'Chat',
                          style: TextStyle(
                            fontSize: 20,
                            fontWeight: FontWeight.w600,
                            color: Color(0xFFFFFFFF),
                          ),
                        ),
                      ],
                    ),
                  ),
                  
                  // Messages
                  Expanded(
                    child: ListView.builder(
                      padding: const EdgeInsets.symmetric(horizontal: 24),
                      itemCount: _messages.length,
                      itemBuilder: (context, index) {
                        return MessageBubble(message: _messages[index]);
                      },
                    ),
                  ),
                  
                  // Input Area
                  Container(
                    padding: const EdgeInsets.all(24),
                    child: Row(
                      children: [
                        Expanded(
                          child: TextField(
                            controller: _messageController,
                            style: const TextStyle(color: Color(0xFFFFFFFF)),
                            decoration: InputDecoration(
                              hintText: 'Ask me anything...',
                              hintStyle: const TextStyle(color: Color(0xFF666666)),
                              border: OutlineInputBorder(
                                borderRadius: BorderRadius.circular(24),
                                borderSide: const BorderSide(color: Color(0xFF333333)),
                              ),
                              enabledBorder: OutlineInputBorder(
                                borderRadius: BorderRadius.circular(24),
                                borderSide: const BorderSide(color: Color(0xFF333333)),
                              ),
                              focusedBorder: OutlineInputBorder(
                                borderRadius: BorderRadius.circular(24),
                                borderSide: const BorderSide(color: Color(0xFF00D4FF)),
                              ),
                              filled: true,
                              fillColor: const Color(0xFF111111),
                              contentPadding: const EdgeInsets.symmetric(
                                horizontal: 20,
                                vertical: 16,
                              ),
                            ),
                            onSubmitted: (_) => _sendMessage(),
                          ),
                        ),
                        const SizedBox(width: 16),
                        Container(
                          decoration: BoxDecoration(
                            color: const Color(0xFF00D4FF),
                            borderRadius: BorderRadius.circular(24),
                          ),
                          child: IconButton(
                            onPressed: _sendMessage,
                            icon: const Icon(
                              Icons.send,
                              color: Color(0xFF000000),
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildSidebarItem(String title) {
    return Container(
      margin: const EdgeInsets.only(bottom: 8),
      child: ListTile(
        title: Text(
          title,
          style: const TextStyle(
            color: Color(0xFFCCCCCC),
            fontSize: 14,
          ),
        ),
        contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
        onTap: () {},
        hoverColor: const Color(0xFF222222),
      ),
    );
  }
}

class ChatMessage {
  final String text;
  final bool isUser;
  final DateTime timestamp;

  ChatMessage({
    required this.text,
    required this.isUser,
    required this.timestamp,
  });
}

class MessageBubble extends StatelessWidget {
  final ChatMessage message;

  const MessageBubble({super.key, required this.message});

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.only(bottom: 24),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisAlignment: message.isUser ? MainAxisAlignment.end : MainAxisAlignment.start,
        children: [
          if (!message.isUser) ...[
            Container(
              width: 40,
              height: 40,
              decoration: BoxDecoration(
                color: const Color(0xFF00D4FF),
                borderRadius: BorderRadius.circular(20),
              ),
              child: const Icon(
                Icons.psychology,
                color: Color(0xFF000000),
                size: 24,
              ),
            ),
            const SizedBox(width: 16),
          ],
          Flexible(
            child: Container(
              constraints: BoxConstraints(
                maxWidth: MediaQuery.of(context).size.width * 0.6,
              ),
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: message.isUser ? const Color(0xFF00D4FF) : const Color(0xFF111111),
                borderRadius: BorderRadius.circular(16),
              ),
              child: Text(
                message.text,
                style: TextStyle(
                  color: message.isUser ? const Color(0xFF000000) : const Color(0xFFFFFFFF),
                  fontSize: 16,
                  height: 1.4,
                ),
              ),
            ),
          ),
          if (message.isUser) ...[
            const SizedBox(width: 16),
            Container(
              width: 40,
              height: 40,
              decoration: BoxDecoration(
                color: const Color(0xFF333333),
                borderRadius: BorderRadius.circular(20),
              ),
              child: const Icon(
                Icons.person,
                color: Color(0xFFFFFFFF),
                size: 24,
              ),
            ),
          ],
        ],
      ),
    );
  }
}
