package com.coconuthealthmonitorNew

import android.app.Application
import com.facebook.react.ReactApplication
import com.facebook.react.ReactNativeHost
import com.facebook.react.ReactPackage
import com.facebook.react.shell.MainReactPackage
import com.facebook.soloader.SoLoader

class MainApplication : Application(), ReactApplication {

  private val mReactNativeHost = object : ReactNativeHost(this) {
    override fun getUseDeveloperSupport(): Boolean = BuildConfig.DEBUG

    override fun getPackages(): List<ReactPackage> {
      val packages = mutableListOf<ReactPackage>()
      // Add the main React package
      packages.add(MainReactPackage(null))

      // Manually add native modules packages
      // TODO: Add autolinking packages here
      packages.add(io.invertase.notifee.NotifeePackage())
      packages.add(com.reactnativecommunity.asyncstorage.AsyncStoragePackage())
      packages.add(io.invertase.firebase.app.ReactNativeFirebaseAppPackage())
      packages.add(io.invertase.firebase.auth.ReactNativeFirebaseAuthPackage())
      packages.add(io.invertase.firebase.messaging.ReactNativeFirebaseMessagingPackage())
      packages.add(com.reactnativegooglesignin.RNGoogleSigninPackage())
      packages.add(com.imagepicker.ImagePickerPackage())
      packages.add(com.th3rdwave.safeareacontext.SafeAreaContextPackage())
      packages.add(com.swmansion.rnscreens.RNScreensPackage())
      packages.add(com.horcrux.svg.SvgPackage())

      return packages
    }

    override fun getJSMainModuleName(): String = "index"
  }

  override val reactNativeHost: ReactNativeHost
    get() = mReactNativeHost

  override fun onCreate() {
    super.onCreate()
    SoLoader.init(this, false)
  }
}
