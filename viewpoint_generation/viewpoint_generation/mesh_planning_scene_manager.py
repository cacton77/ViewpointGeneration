#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
from geometry_msgs.msg import Pose, Point
from shape_msgs.msg import Mesh, MeshTriangle
from moveit_msgs.msg import CollisionObject, AttachedCollisionObject
from moveit_msgs.srv import ApplyPlanningScene, GetPlanningScene
import moveit_py
from moveit_py import MoveItPy
from moveit_py.core import PlanningScene
import open3d as o3d

class MeshPlanningSceneManager(Node):
    def __init__(self):
        super().__init__('mesh_planning_scene_manager')
        
        # Initialize MoveIt
        self.moveit = MoveItPy(node_name="moveit_py_planning_scene")
        self.planning_scene_monitor = self.moveit.get_planning_scene_monitor()
        
        # Get planning scene
        self.planning_scene = self.planning_scene_monitor.read_only
        
        # Service clients for planning scene
        self.apply_scene_client = self.create_client(
            ApplyPlanningScene, 
            '/apply_planning_scene'
        )
        
        self.get_scene_client = self.create_client(
            GetPlanningScene,
            '/get_planning_scene'
        )
        
        # Wait for services
        self.get_logger().info("Waiting for planning scene services...")
        self.apply_scene_client.wait_for_service()
        self.get_scene_client.wait_for_service()
        self.get_logger().info("Planning scene services ready!")

    def load_mesh_from_file(self, mesh_file_path):
        """Load a triangle mesh from file using Open3D"""
        try:
            mesh = o3d.io.read_triangle_mesh(mesh_file_path)
            
            if len(mesh.vertices) == 0:
                self.get_logger().error(f"Failed to load mesh from {mesh_file_path}")
                return None
                
            self.get_logger().info(f"Loaded mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
            return mesh
            
        except Exception as e:
            self.get_logger().error(f"Error loading mesh: {str(e)}")
            return None

    def create_mesh_message(self, o3d_mesh):
        """Convert Open3D mesh to ROS Mesh message"""
        mesh_msg = Mesh()
        
        # Add vertices
        vertices = np.asarray(o3d_mesh.vertices)
        for vertex in vertices:
            point = Point()
            point.x = float(vertex[0])
            point.y = float(vertex[1]) 
            point.z = float(vertex[2])
            mesh_msg.vertices.append(point)
        
        # Add triangles
        triangles = np.asarray(o3d_mesh.triangles)
        for triangle in triangles:
            mesh_triangle = MeshTriangle()
            mesh_triangle.vertex_indices = [int(triangle[0]), int(triangle[1]), int(triangle[2])]
            mesh_msg.triangles.append(mesh_triangle)
            
        self.get_logger().info(f"Created mesh message: {len(mesh_msg.vertices)} vertices, {len(mesh_msg.triangles)} triangles")
        return mesh_msg

    def create_collision_object_from_mesh(self, mesh_file_path, object_id, pose=None):
        """Create a CollisionObject from a mesh file"""
        
        # Load mesh
        o3d_mesh = self.load_mesh_from_file(mesh_file_path)
        if o3d_mesh is None:
            return None
            
        # Convert to ROS message
        mesh_msg = self.create_mesh_message(o3d_mesh)
        
        # Create collision object
        collision_object = CollisionObject()
        collision_object.header.frame_id = "object_frame"  # Your specified frame
        collision_object.header.stamp = self.get_clock().now().to_msg()
        collision_object.id = object_id
        
        # Set pose (default to identity if not provided)
        if pose is None:
            pose = Pose()
            pose.orientation.w = 1.0  # Identity quaternion
            
        collision_object.meshes = [mesh_msg]
        collision_object.mesh_poses = [pose]
        collision_object.operation = CollisionObject.ADD
        
        return collision_object

    def add_mesh_to_planning_scene(self, mesh_file_path, object_id="mesh_object", pose=None):
        """Add a triangle mesh to the planning scene"""
        
        # Create collision object
        collision_object = self.create_collision_object_from_mesh(mesh_file_path, object_id, pose)
        if collision_object is None:
            self.get_logger().error("Failed to create collision object")
            return False
            
        # Apply to planning scene
        return self.apply_collision_object(collision_object)

    def attach_mesh_to_link(self, mesh_file_path, object_id, link_name="object_frame", 
                           touch_links=None, pose=None):
        """Attach a triangle mesh to a specific link"""
        
        # Create collision object first
        collision_object = self.create_collision_object_from_mesh(mesh_file_path, object_id, pose)
        if collision_object is None:
            self.get_logger().error("Failed to create collision object")
            return False
        
        # Create attached collision object
        attached_object = AttachedCollisionObject()
        attached_object.link_name = link_name
        attached_object.object = collision_object
        
        # Set touch links (links that can touch this object without collision)
        if touch_links is None:
            touch_links = [link_name]  # Default to the attached link
        attached_object.touch_links = touch_links
        
        # Apply to planning scene
        return self.apply_attached_object(attached_object)

    def apply_collision_object(self, collision_object):
        """Apply a collision object to the planning scene"""
        try:
            # Create planning scene diff
            from moveit_msgs.msg import PlanningScene
            planning_scene_diff = PlanningScene()
            planning_scene_diff.world.collision_objects.append(collision_object)
            planning_scene_diff.is_diff = True
            
            # Apply the scene
            request = ApplyPlanningScene.Request()
            request.scene = planning_scene_diff
            
            future = self.apply_scene_client.call_async(request)
            rclpy.spin_until_future_complete(self, future)
            
            if future.result().success:
                self.get_logger().info(f"Successfully added collision object: {collision_object.id}")
                return True
            else:
                self.get_logger().error(f"Failed to add collision object: {collision_object.id}")
                return False
                
        except Exception as e:
            self.get_logger().error(f"Error applying collision object: {str(e)}")
            return False

    def apply_attached_object(self, attached_object):
        """Apply an attached collision object to the planning scene"""
        try:
            # Create planning scene diff
            from moveit_msgs.msg import PlanningScene
            planning_scene_diff = PlanningScene()
            planning_scene_diff.robot_state.attached_collision_objects.append(attached_object)
            planning_scene_diff.is_diff = True
            
            # Apply the scene
            request = ApplyPlanningScene.Request()
            request.scene = planning_scene_diff
            
            future = self.apply_scene_client.call_async(request)
            rclpy.spin_until_future_complete(self, future)
            
            if future.result().success:
                self.get_logger().info(f"Successfully attached object: {attached_object.object.id} to {attached_object.link_name}")
                return True
            else:
                self.get_logger().error(f"Failed to attach object: {attached_object.object.id}")
                return False
                
        except Exception as e:
            self.get_logger().error(f"Error applying attached object: {str(e)}")
            return False

    def remove_object_from_scene(self, object_id):
        """Remove an object from the planning scene"""
        try:
            collision_object = CollisionObject()
            collision_object.id = object_id
            collision_object.operation = CollisionObject.REMOVE
            
            return self.apply_collision_object(collision_object)
            
        except Exception as e:
            self.get_logger().error(f"Error removing object: {str(e)}")
            return False

    def detach_object_from_link(self, object_id, link_name="object_frame"):
        """Detach an object from a link"""
        try:
            attached_object = AttachedCollisionObject()
            attached_object.object.id = object_id
            attached_object.link_name = link_name
            attached_object.object.operation = CollisionObject.REMOVE
            
            return self.apply_attached_object(attached_object)
            
        except Exception as e:
            self.get_logger().error(f"Error detaching object: {str(e)}")
            return False

    def create_simple_mesh_example(self):
        """Create a simple triangle mesh for testing"""
        import open3d as o3d
        
        # Create a simple box mesh
        mesh = o3d.geometry.TriangleMesh.create_box(width=0.1, height=0.1, depth=0.1)
        mesh.translate([-0.05, -0.05, -0.05])  # Center it
        
        # Convert to ROS message
        return self.create_mesh_message(mesh)

    def add_simple_mesh_to_object_frame(self):
        """Add a simple mesh directly to object_frame (example usage)"""
        
        # Create simple mesh
        mesh_msg = self.create_simple_mesh_example()
        
        # Create collision object
        collision_object = CollisionObject()
        collision_object.header.frame_id = "object_frame"
        collision_object.header.stamp = self.get_clock().now().to_msg()
        collision_object.id = "simple_box"
        
        # Set pose
        pose = Pose()
        pose.orientation.w = 1.0
        
        collision_object.meshes = [mesh_msg]
        collision_object.mesh_poses = [pose]
        collision_object.operation = CollisionObject.ADD
        
        return self.apply_collision_object(collision_object)

def main():
    rclpy.init()
    
    # Create the planning scene manager
    manager = MeshPlanningSceneManager()
    
    try:
        # Example 1: Add mesh from file to planning scene
        mesh_file = "/path/to/your/mesh.stl"  # Replace with your mesh file
        success = manager.add_mesh_to_planning_scene(mesh_file, "my_mesh_object")
        
        if success:
            manager.get_logger().info("Mesh added to planning scene successfully!")
        
        # Example 2: Attach mesh to object_frame link
        from geometry_msgs.msg import Pose
        pose = Pose()
        pose.position.z = 0.1  # 10cm above the link
        pose.orientation.w = 1.0
        
        success = manager.attach_mesh_to_link(
            mesh_file, 
            "attached_mesh", 
            link_name="object_frame",
            pose=pose
        )
        
        if success:
            manager.get_logger().info("Mesh attached to object_frame successfully!")
        
        # Example 3: Add simple test mesh
        success = manager.add_simple_mesh_to_object_frame()
        if success:
            manager.get_logger().info("Simple mesh added successfully!")
        
        # Keep node alive
        rclpy.spin(manager)
        
    except KeyboardInterrupt:
        manager.get_logger().info("Shutting down...")
    finally:
        manager.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

# Alternative: Direct MoveIt Python API approach (if available)
class DirectMoveItPyApproach:
    """Alternative approach using direct moveit_py APIs"""
    
    def __init__(self):
        # Initialize moveit_py
        self.moveit = MoveItPy(node_name="direct_moveit_py")
        self.planning_scene_monitor = self.moveit.get_planning_scene_monitor()
    
    def add_mesh_collision_object(self, mesh_file_path, object_id, frame_id="object_frame"):
        """Add mesh using direct moveit_py planning scene interface"""
        
        # Get planning scene
        with self.planning_scene_monitor.read_write() as scene:
            # Load mesh
            mesh = o3d.io.read_triangle_mesh(mesh_file_path)
            
            # Convert mesh data
            vertices = np.asarray(mesh.vertices).tolist()
            triangles = np.asarray(mesh.triangles).tolist()
            
            # Add collision object (this syntax may vary based on moveit_py version)
            scene.add_collision_mesh(
                object_id,
                frame_id,
                vertices,
                triangles
            )
            
        return True

# Usage examples for different scenarios:

def example_usage():
    """Examples of different ways to add meshes"""
    
    rclpy.init()
    manager = MeshPlanningSceneManager()
    
    # Scenario 1: Static obstacle in the scene
    manager.add_mesh_to_planning_scene(
        "/path/to/obstacle.stl", 
        "static_obstacle"
    )
    
    # Scenario 2: Tool attached to end-effector
    tool_pose = Pose()
    tool_pose.position.z = 0.15  # 15cm offset
    tool_pose.orientation.w = 1.0
    
    manager.attach_mesh_to_link(
        "/path/to/tool.stl",
        "end_effector_tool",
        link_name="object_frame",
        touch_links=["object_frame", "gripper_link"],  # Links that can touch
        pose=tool_pose
    )
    
    # Scenario 3: Object to be manipulated
    object_pose = Pose()
    object_pose.position.x = 0.3
    object_pose.position.y = 0.2
    object_pose.position.z = 0.8
    object_pose.orientation.w = 1.0
    
    manager.add_mesh_to_planning_scene(
        "/path/to/object.stl",
        "target_object", 
        pose=object_pose
    )
    
    rclpy.shutdown()