const {getDefaultConfig, mergeConfig} = require('@react-native/metro-config');
const path = require('path');

/**
 * Metro configuration
 * https://reactnative.dev/docs/metro
 *
 * @type {import('@react-native/metro-config').MetroConfig}
 */
const config = {
  resolver: {
    nodeModulesPaths: [],
    blockList: [
      new RegExp(path.resolve(__dirname, 'backend', 'node_modules').replace(/[/\\]/g, '[/\\\\]') + '.*'),
      new RegExp(path.resolve(__dirname, 'ml').replace(/[/\\]/g, '[/\\\\]') + '.*'),
    ],
    unstable_enablePackageExports: true,
  },
  watcher: {
    watchman: {
      enabled: false,
    },
    healthCheck: {
      enabled: false,
    },
  },
};

module.exports = mergeConfig(getDefaultConfig(__dirname), config);
