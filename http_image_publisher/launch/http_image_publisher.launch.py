# launch/camera_with_transport.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        # HTTP Image Publisher
        Node(
            package='http_image_publisher',
            executable='http_image_publisher',
            name='camera_publisher',
            parameters=[{
                'stream_url': 'http://192.168.0.92:5000/video_feed',
                'base_topic': 'camera',
                'publish_rate': 30.0
            }],
            # arguments=['--ros-args', '--log-level', 'DEBUG']
        ),

        # Compressed republisher
        Node(
            package='image_transport',
            executable='republish',
            name='compressed_republisher',
            arguments=['raw', 'compressed'],
            remappings=[
                ('in', 'camera/image_raw'),
                ('out/compressed', 'camera/image_raw/compressed')
            ]
        ),

        # Theora republisher with explicit parameters
        Node(
            package='image_transport',
            executable='republish',
            name='theora_republisher',
            arguments=['raw', 'theora'],
            parameters=[{
                'theora.optimize_for': 0,
                'theora.quality': 40,
                'theora.bitrate': 800000,
                'theora.keyframe_frequency': 30
            }],
            remappings=[
                ('in', 'camera/image_raw'),
                ('out/theora', 'camera/image_raw/theora')
            ]
        )
    ])
