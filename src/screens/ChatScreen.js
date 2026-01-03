import React, {useState, useRef, useEffect} from 'react';
import {
  View,
  Text,
  StyleSheet,
  TextInput,
  TouchableOpacity,
  FlatList,
  KeyboardAvoidingView,
  Platform,
  ActivityIndicator,
  SafeAreaView,
  Keyboard,
} from 'react-native';
import {useLanguage} from '../context/LanguageContext';
import {sendChatMessage, checkChatHealth} from '../services/chatApi';

const ChatScreen = ({navigation}) => {
  const {t} = useLanguage();
  const [messages, setMessages] = useState([]);
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [chatStatus, setChatStatus] = useState('checking');
  const flatListRef = useRef(null);

  // Welcome message
  useEffect(() => {
    const welcomeMessage = {
      id: 'welcome',
      role: 'assistant',
      content: t('chat.welcomeMessage') ||
        "Hello! I'm your Coconut Health Assistant. I can help you with:\n\n" +
        "🐛 Pest treatment (mite, caterpillar, white fly)\n" +
        "🌴 Disease management\n" +
        "💡 Farming tips\n\n" +
        "How can I help you today?",
      timestamp: new Date().toISOString(),
    };
    setMessages([welcomeMessage]);
    checkStatus();
  }, []);

  const checkStatus = async () => {
    try {
      const health = await checkChatHealth();
      setChatStatus(health.success ? 'online' : 'offline');
    } catch (error) {
      setChatStatus('offline');
    }
  };

  const sendMessage = async () => {
    if (!inputText.trim() || isLoading) return;

    const userMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: inputText.trim(),
      timestamp: new Date().toISOString(),
    };

    // Add user message to chat
    setMessages(prev => [...prev, userMessage]);
    setInputText('');
    setIsLoading(true);
    Keyboard.dismiss();

    try {
      // Prepare chat history for API (exclude welcome message)
      const history = messages
        .filter(m => m.id !== 'welcome')
        .map(m => ({role: m.role, content: m.content}));

      const response = await sendChatMessage(inputText.trim(), history);

      if (response.success) {
        const assistantMessage = {
          id: (Date.now() + 1).toString(),
          role: 'assistant',
          content: response.response,
          timestamp: new Date().toISOString(),
        };
        setMessages(prev => [...prev, assistantMessage]);
      } else {
        // Error message
        const errorMessage = {
          id: (Date.now() + 1).toString(),
          role: 'assistant',
          content: `Sorry, I couldn't process your request. ${response.error || 'Please try again.'}`,
          timestamp: new Date().toISOString(),
          isError: true,
        };
        setMessages(prev => [...prev, errorMessage]);
      }
    } catch (error) {
      const errorMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: 'Sorry, something went wrong. Please check your connection and try again.',
        timestamp: new Date().toISOString(),
        isError: true,
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const renderMessage = ({item}) => {
    const isUser = item.role === 'user';
    const isError = item.isError;

    return (
      <View
        style={[
          styles.messageContainer,
          isUser ? styles.userMessageContainer : styles.assistantMessageContainer,
        ]}>
        {!isUser && (
          <View style={styles.avatarContainer}>
            <Text style={styles.avatarText}>🥥</Text>
          </View>
        )}
        <View
          style={[
            styles.messageBubble,
            isUser ? styles.userBubble : styles.assistantBubble,
            isError && styles.errorBubble,
          ]}>
          <Text
            style={[
              styles.messageText,
              isUser ? styles.userMessageText : styles.assistantMessageText,
            ]}>
            {item.content}
          </Text>
          <Text style={styles.timestampText}>
            {new Date(item.timestamp).toLocaleTimeString([], {
              hour: '2-digit',
              minute: '2-digit',
            })}
          </Text>
        </View>
        {isUser && (
          <View style={styles.avatarContainer}>
            <Text style={styles.avatarText}>👤</Text>
          </View>
        )}
      </View>
    );
  };

  const renderTypingIndicator = () => {
    if (!isLoading) return null;

    return (
      <View style={[styles.messageContainer, styles.assistantMessageContainer]}>
        <View style={styles.avatarContainer}>
          <Text style={styles.avatarText}>🥥</Text>
        </View>
        <View style={[styles.messageBubble, styles.assistantBubble, styles.typingBubble]}>
          <View style={styles.typingDots}>
            <View style={[styles.dot, styles.dot1]} />
            <View style={[styles.dot, styles.dot2]} />
            <View style={[styles.dot, styles.dot3]} />
          </View>
        </View>
      </View>
    );
  };

  const quickQuestions = [
    {id: '1', text: 'How to treat mite?', emoji: '🐛'},
    {id: '2', text: 'Caterpillar treatment', emoji: '🐛'},
    {id: '3', text: 'White fly control', emoji: '🦟'},
    {id: '4', text: 'Coconut care tips', emoji: '🌴'},
  ];

  const renderQuickQuestions = () => {
    if (messages.length > 1) return null;

    return (
      <View style={styles.quickQuestionsContainer}>
        <Text style={styles.quickQuestionsTitle}>
          {t('chat.quickQuestions') || 'Quick Questions:'}
        </Text>
        <View style={styles.quickQuestionsRow}>
          {quickQuestions.map(q => (
            <TouchableOpacity
              key={q.id}
              style={styles.quickQuestionButton}
              onPress={() => setInputText(q.text)}>
              <Text style={styles.quickQuestionEmoji}>{q.emoji}</Text>
              <Text style={styles.quickQuestionText}>{q.text}</Text>
            </TouchableOpacity>
          ))}
        </View>
      </View>
    );
  };

  return (
    <SafeAreaView style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity
          style={styles.backButton}
          onPress={() => navigation.goBack()}>
          <Text style={styles.backButtonText}>← {t('common.back') || 'Back'}</Text>
        </TouchableOpacity>
        <View style={styles.headerTitleContainer}>
          <Text style={styles.headerTitle}>🥥 {t('chat.title') || 'Coconut Assistant'}</Text>
          <View
            style={[
              styles.statusBadge,
              chatStatus === 'online' ? styles.statusOnline : styles.statusOffline,
            ]}>
            <Text style={styles.statusText}>
              {chatStatus === 'online' ? 'AI Online' : 'Offline'}
            </Text>
          </View>
        </View>
        <View style={styles.headerSpacer} />
      </View>

      {/* Chat Messages */}
      <KeyboardAvoidingView
        style={styles.chatContainer}
        behavior={Platform.OS === 'ios' ? 'padding' : undefined}
        keyboardVerticalOffset={Platform.OS === 'ios' ? 90 : 0}>
        <FlatList
          ref={flatListRef}
          data={messages}
          renderItem={renderMessage}
          keyExtractor={item => item.id}
          contentContainerStyle={styles.messagesList}
          onContentSizeChange={() =>
            flatListRef.current?.scrollToEnd({animated: true})
          }
          onLayout={() => flatListRef.current?.scrollToEnd({animated: true})}
          ListFooterComponent={renderTypingIndicator}
          ListHeaderComponent={renderQuickQuestions}
        />

        {/* Input Area */}
        <View style={styles.inputContainer}>
          <TextInput
            style={styles.textInput}
            value={inputText}
            onChangeText={setInputText}
            placeholder={t('chat.placeholder') || 'Ask about coconut health...'}
            placeholderTextColor="#999"
            multiline
            maxLength={500}
            editable={!isLoading && chatStatus === 'online'}
          />
          <TouchableOpacity
            style={[
              styles.sendButton,
              (!inputText.trim() || isLoading || chatStatus !== 'online') &&
                styles.sendButtonDisabled,
            ]}
            onPress={sendMessage}
            disabled={!inputText.trim() || isLoading || chatStatus !== 'online'}>
            {isLoading ? (
              <ActivityIndicator color="#fff" size="small" />
            ) : (
              <Text style={styles.sendButtonText}>➤</Text>
            )}
          </TouchableOpacity>
        </View>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 15,
    paddingVertical: 12,
    backgroundColor: '#fff',
    borderBottomWidth: 1,
    borderBottomColor: '#e0e0e0',
    elevation: 2,
    shadowColor: '#000',
    shadowOffset: {width: 0, height: 1},
    shadowOpacity: 0.1,
    shadowRadius: 2,
  },
  backButton: {
    padding: 5,
  },
  backButtonText: {
    fontSize: 16,
    color: '#2e7d32',
    fontWeight: '600',
  },
  headerTitleContainer: {
    alignItems: 'center',
  },
  headerTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
  },
  statusBadge: {
    paddingHorizontal: 8,
    paddingVertical: 2,
    borderRadius: 10,
    marginTop: 4,
  },
  statusOnline: {
    backgroundColor: '#e8f5e9',
  },
  statusOffline: {
    backgroundColor: '#ffebee',
  },
  statusText: {
    fontSize: 10,
    fontWeight: '600',
    color: '#2e7d32',
  },
  headerSpacer: {
    width: 60,
  },
  chatContainer: {
    flex: 1,
  },
  messagesList: {
    padding: 15,
    paddingBottom: 10,
  },
  messageContainer: {
    flexDirection: 'row',
    marginBottom: 12,
    alignItems: 'flex-end',
  },
  userMessageContainer: {
    justifyContent: 'flex-end',
  },
  assistantMessageContainer: {
    justifyContent: 'flex-start',
  },
  avatarContainer: {
    width: 36,
    height: 36,
    borderRadius: 18,
    backgroundColor: '#e8f5e9',
    justifyContent: 'center',
    alignItems: 'center',
    marginHorizontal: 8,
  },
  avatarText: {
    fontSize: 18,
  },
  messageBubble: {
    maxWidth: '70%',
    padding: 12,
    borderRadius: 18,
  },
  userBubble: {
    backgroundColor: '#2e7d32',
    borderBottomRightRadius: 4,
  },
  assistantBubble: {
    backgroundColor: '#fff',
    borderBottomLeftRadius: 4,
    borderWidth: 1,
    borderColor: '#e0e0e0',
  },
  errorBubble: {
    backgroundColor: '#ffebee',
    borderColor: '#ffcdd2',
  },
  messageText: {
    fontSize: 15,
    lineHeight: 22,
  },
  userMessageText: {
    color: '#fff',
  },
  assistantMessageText: {
    color: '#333',
  },
  timestampText: {
    fontSize: 10,
    color: '#999',
    marginTop: 4,
    alignSelf: 'flex-end',
  },
  typingBubble: {
    paddingVertical: 15,
    paddingHorizontal: 20,
  },
  typingDots: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  dot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: '#2e7d32',
    marginHorizontal: 2,
    opacity: 0.4,
  },
  dot1: {
    opacity: 1,
  },
  dot2: {
    opacity: 0.7,
  },
  dot3: {
    opacity: 0.4,
  },
  quickQuestionsContainer: {
    marginBottom: 15,
  },
  quickQuestionsTitle: {
    fontSize: 14,
    color: '#666',
    marginBottom: 10,
    textAlign: 'center',
  },
  quickQuestionsRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'center',
  },
  quickQuestionButton: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#fff',
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 20,
    margin: 4,
    borderWidth: 1,
    borderColor: '#e0e0e0',
  },
  quickQuestionEmoji: {
    fontSize: 14,
    marginRight: 6,
  },
  quickQuestionText: {
    fontSize: 13,
    color: '#333',
  },
  inputContainer: {
    flexDirection: 'row',
    alignItems: 'flex-end',
    padding: 10,
    backgroundColor: '#fff',
    borderTopWidth: 1,
    borderTopColor: '#e0e0e0',
  },
  textInput: {
    flex: 1,
    minHeight: 44,
    maxHeight: 100,
    backgroundColor: '#f5f5f5',
    borderRadius: 22,
    paddingHorizontal: 18,
    paddingVertical: 10,
    fontSize: 15,
    color: '#333',
    marginRight: 10,
  },
  sendButton: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: '#2e7d32',
    justifyContent: 'center',
    alignItems: 'center',
  },
  sendButtonDisabled: {
    backgroundColor: '#ccc',
  },
  sendButtonText: {
    fontSize: 20,
    color: '#fff',
  },
});

export default ChatScreen;
