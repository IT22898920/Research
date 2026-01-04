import React, {useState, useEffect} from 'react';
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  StyleSheet,
  KeyboardAvoidingView,
  Platform,
  Alert,
  ActivityIndicator,
} from 'react-native';
import {configureGoogleSignIn, signInWithGoogle} from '../config/googleAuth';
import {authAPI} from '../services/api';
import {useLanguage} from '../context/LanguageContext';

export default function LoginScreen({navigation}) {
  console.log('=== LoginScreen rendering ===');
  const {t} = useLanguage();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isGoogleLoading, setIsGoogleLoading] = useState(false);
  const [isCheckingLogin, setIsCheckingLogin] = useState(true);
  console.log('=== LoginScreen isCheckingLogin:', isCheckingLogin, '===');

  useEffect(() => {
    configureGoogleSignIn();
    checkExistingLogin();
  }, []);

  // Navigate based on user role
  const navigateToDashboard = (user) => {
    if (user.role === 'admin') {
      navigation.replace('AdminDashboard', {user});
    } else {
      navigation.replace('Dashboard', {user});
    }
  };

  // Check if user is already logged in
  const checkExistingLogin = async () => {
    console.log('=== checkExistingLogin started ===');
    try {
      console.log('=== Calling authAPI.isLoggedIn() ===');
      const isLoggedIn = await authAPI.isLoggedIn();
      console.log('=== isLoggedIn result:', isLoggedIn, '===');
      if (isLoggedIn) {
        const user = await authAPI.getStoredUser();
        if (user) {
          navigateToDashboard(user);
          return;
        }
      }
    } catch (error) {
      console.log('No existing login, error:', error);
    }
    console.log('=== setIsCheckingLogin(false) ===');
    setIsCheckingLogin(false);
  };

  // Email/Password Login
  const handleLogin = async () => {
    if (!email || !password) {
      Alert.alert('Error', 'Please enter email and password');
      return;
    }

    setIsLoading(true);
    try {
      const response = await authAPI.login(email, password);

      if (response.success) {
        navigateToDashboard(response.data.user);
      } else {
        Alert.alert('Login Failed', response.message || 'Invalid credentials');
      }
    } catch (error) {
      Alert.alert('Login Failed', error.message || 'Something went wrong');
    } finally {
      setIsLoading(false);
    }
  };

  // Google Sign-In
  const handleGoogleSignIn = async () => {
    setIsGoogleLoading(true);
    try {
      // First, authenticate with Google/Firebase
      const googleResult = await signInWithGoogle();

      if (googleResult.success) {
        // Then, register/login with our backend
        const backendResponse = await authAPI.googleAuth({
          email: googleResult.data.email,
          displayName: googleResult.data.displayName,
          photoURL: googleResult.data.photoURL,
          firebaseUid: googleResult.data.uid,
        });

        if (backendResponse.success) {
          navigateToDashboard(backendResponse.data.user);
        } else {
          Alert.alert('Error', backendResponse.message || 'Backend authentication failed');
        }
      } else {
        Alert.alert('Sign-In Failed', googleResult.error || 'Google sign-in failed');
      }
    } catch (error) {
      Alert.alert('Error', error.message || 'Something went wrong');
    } finally {
      setIsGoogleLoading(false);
    }
  };

  // Show loading while checking existing login
  if (isCheckingLogin) {
    return (
      <View style={[styles.container, styles.centerContent]}>
        <ActivityIndicator size="large" color="#2e7d32" />
      </View>
    );
  }

  return (
    <KeyboardAvoidingView
      behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
      style={styles.container}>
      <View style={styles.logoContainer}>
        <Text style={styles.logoText}>🌴</Text>
        <Text style={styles.title}>{t('common.appName')}</Text>
        <Text style={styles.subtitle}>AI-Powered Drone Monitoring System</Text>
      </View>

      <View style={styles.formContainer}>
        <TextInput
          style={styles.input}
          placeholder={t('auth.email')}
          placeholderTextColor="#999"
          value={email}
          onChangeText={setEmail}
          keyboardType="email-address"
          autoCapitalize="none"
          editable={!isLoading && !isGoogleLoading}
        />

        <TextInput
          style={styles.input}
          placeholder={t('auth.password')}
          placeholderTextColor="#999"
          value={password}
          onChangeText={text => setPassword(text)}
          secureTextEntry={true}
          autoCapitalize="none"
          autoCorrect={false}
          textContentType="password"
          editable={!isLoading && !isGoogleLoading}
        />

        <TouchableOpacity
          style={[styles.button, isLoading && styles.buttonDisabled]}
          onPress={handleLogin}
          disabled={isLoading || isGoogleLoading}>
          {isLoading ? (
            <ActivityIndicator color="#fff" />
          ) : (
            <Text style={styles.buttonText}>{t('auth.login')}</Text>
          )}
        </TouchableOpacity>

        <View style={styles.dividerContainer}>
          <View style={styles.dividerLine} />
          <Text style={styles.dividerText}>OR</Text>
          <View style={styles.dividerLine} />
        </View>

        <TouchableOpacity
          style={[styles.googleButton, isGoogleLoading && styles.buttonDisabled]}
          onPress={handleGoogleSignIn}
          disabled={isLoading || isGoogleLoading}>
          {isGoogleLoading ? (
            <ActivityIndicator color="#333" />
          ) : (
            <>
              <Text style={styles.googleIcon}>G</Text>
              <Text style={styles.googleButtonText}>{t('auth.googleSignIn')}</Text>
            </>
          )}
        </TouchableOpacity>

        <TouchableOpacity
          style={styles.linkContainer}
          onPress={() => navigation.navigate('Signup')}
          disabled={isLoading || isGoogleLoading}>
          <Text style={styles.linkText}>
            {t('auth.noAccount')} <Text style={styles.linkBold}>{t('auth.signup')}</Text>
          </Text>
        </TouchableOpacity>
      </View>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  centerContent: {
    justifyContent: 'center',
    alignItems: 'center',
  },
  logoContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingTop: 50,
  },
  logoText: {
    fontSize: 80,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#2e7d32',
    marginTop: 10,
  },
  subtitle: {
    fontSize: 14,
    color: '#666',
    marginTop: 5,
  },
  formContainer: {
    flex: 2,
    paddingHorizontal: 30,
    paddingTop: 30,
  },
  input: {
    backgroundColor: '#fff',
    borderWidth: 1,
    borderColor: '#ddd',
    padding: 15,
    marginBottom: 15,
    borderRadius: 10,
    fontSize: 16,
    color: '#333',
  },
  button: {
    backgroundColor: '#2e7d32',
    padding: 15,
    borderRadius: 10,
    alignItems: 'center',
    marginTop: 10,
  },
  buttonDisabled: {
    opacity: 0.7,
  },
  buttonText: {
    color: 'white',
    fontWeight: 'bold',
    fontSize: 18,
  },
  dividerContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginVertical: 20,
  },
  dividerLine: {
    flex: 1,
    height: 1,
    backgroundColor: '#ddd',
  },
  dividerText: {
    marginHorizontal: 15,
    color: '#999',
    fontSize: 14,
  },
  googleButton: {
    backgroundColor: '#fff',
    padding: 15,
    borderRadius: 10,
    alignItems: 'center',
    flexDirection: 'row',
    justifyContent: 'center',
    borderWidth: 1,
    borderColor: '#ddd',
  },
  googleIcon: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#4285F4',
    marginRight: 10,
  },
  googleButtonText: {
    color: '#333',
    fontWeight: '600',
    fontSize: 16,
  },
  linkContainer: {
    marginTop: 25,
    alignItems: 'center',
  },
  linkText: {
    color: '#666',
    fontSize: 14,
  },
  linkBold: {
    color: '#2e7d32',
    fontWeight: 'bold',
  },
});
