class NavigationData {
  final String timestamp;
  final String imageBase64;
  final SensorData sensors;
  final DeviceOrientation deviceOrientation;

  NavigationData({
    required this.timestamp,
    required this.imageBase64,
    required this.sensors,
    required this.deviceOrientation,
  });

  Map<String, dynamic> toJson() {
    return {
      'timestamp': timestamp,
      'image_base64': imageBase64,
      'sensors': sensors.toJson(),
      'device_orientation': deviceOrientation.toJson(),
    };
  }
}

class SensorData {
  final AccelerometerData accelerometer;
  final GyroscopeData gyroscope;
  final MagnetometerData magnetometer;

  SensorData({
    required this.accelerometer,
    required this.gyroscope,
    required this.magnetometer,
  });

  Map<String, dynamic> toJson() {
    return {
      'accelerometer': accelerometer.toJson(),
      'gyroscope': gyroscope.toJson(),
      'magnetometer': magnetometer.toJson(),
    };
  }
}

class AccelerometerData {
  final double x, y, z;

  AccelerometerData({required this.x, required this.y, required this.z});

  Map<String, dynamic> toJson() {
    return {'x': x, 'y': y, 'z': z};
  }
}

class GyroscopeData {
  final double x, y, z;

  GyroscopeData({required this.x, required this.y, required this.z});

  Map<String, dynamic> toJson() {
    return {'x': x, 'y': y, 'z': z};
  }
}

class MagnetometerData {
  final double x, y, z;

  MagnetometerData({required this.x, required this.y, required this.z});

  Map<String, dynamic> toJson() {
    return {'x': x, 'y': y, 'z': z};
  }
}

class DeviceOrientation {
  final double roll, pitch;

  DeviceOrientation({required this.roll, required this.pitch});

  Map<String, dynamic> toJson() {
    return {'roll': roll, 'pitch': pitch};
  }
}