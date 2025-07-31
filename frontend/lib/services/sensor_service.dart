import 'dart:math'; // Important: Add this import for math functions
import 'dart:async';
import 'package:sensors_plus/sensors_plus.dart';
import '../models/navigation_data.dart';

class SensorService {
  StreamSubscription<UserAccelerometerEvent>? _accelerometerSubscription;
  StreamSubscription<GyroscopeEvent>? _gyroscopeSubscription;
  StreamSubscription<MagnetometerEvent>? _magnetometerSubscription;

  AccelerometerData? _lastAccelerometerData;
  GyroscopeData? _lastGyroscopeData;
  MagnetometerData? _lastMagnetometerData;

  void startListening() {
    _accelerometerSubscription = userAccelerometerEvents.listen((event) {
      _lastAccelerometerData = AccelerometerData(x: event.x, y: event.y, z: event.z);
    });

    _gyroscopeSubscription = gyroscopeEvents.listen((event) {
      _lastGyroscopeData = GyroscopeData(x: event.x, y: event.y, z: event.z);
    });

    _magnetometerSubscription = magnetometerEvents.listen((event) {
      _lastMagnetometerData = MagnetometerData(x: event.x, y: event.y, z: event.z);
    });
  }

  void stopListening() {
    _accelerometerSubscription?.cancel();
    _gyroscopeSubscription?.cancel();
    _magnetometerSubscription?.cancel();
  }

  SensorData getCurrentSensorData() {
    return SensorData(
      accelerometer: _lastAccelerometerData ?? AccelerometerData(x: 0, y: 0, z: 0),
      gyroscope: _lastGyroscopeData ?? GyroscopeData(x: 0, y: 0, z: 0),
      magnetometer: _lastMagnetometerData ?? MagnetometerData(x: 0, y: 0, z: 0),
    );
  }

  DeviceOrientation calculateOrientation() {
    // Simplified orientation calculation
    final accel = _lastAccelerometerData;
    if (accel == null) return DeviceOrientation(roll: 0, pitch: 0);

    // Calculate roll and pitch from accelerometer data
    double roll = atan2(accel.y, accel.z) * 180 / pi;
    double pitch = atan2(-accel.x, sqrt(accel.y * accel.y + accel.z * accel.z)) * 180 / pi;

    return DeviceOrientation(roll: roll, pitch: pitch);
  }
}