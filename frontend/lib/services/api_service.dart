import 'dart:convert';
import 'package:http/http.dart' as http;
import '../models/navigation_data.dart';

class ApiService {
  static const String baseUrl = 'http://localhost:8000'; // Update for your backend

  Future<Map<String, dynamic>> sendNavigationData(NavigationData data) async {
    try {
      final response = await http.post(
        Uri.parse('$baseUrl/navigate'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode(data.toJson()),
      );

      if (response.statusCode == 200) {
        return jsonDecode(response.body);
      } else {
        throw Exception('Failed to process navigation data: ${response.statusCode}');
      }
    } catch (e) {
      throw Exception('Network error: $e');
    }
  }
}