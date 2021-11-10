/*Created by Daila De Santis and Mahdieh Nejati
Script for simulating different multi-stage tajectories given start and goal poses. 
*/

#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>

#include <moveit_msgs/DisplayRobotState.h>
#include <moveit_msgs/DisplayTrajectory.h>

#include <moveit_msgs/AttachedCollisionObject.h>
#include <moveit_msgs/CollisionObject.h>
#include <std_msgs/String.h>
#include <vector>

#include <iostream>

//#include <tf/Pose.h>

#include <kinova_driver/kinova_ros_types.h>

namespace JacoBoMI{
class trajectory_simulator
{
    ros::NodeHandle nh_;

    tf::Pose Home_;
    tf::Pose Start_; 
    tf::Pose Goal_;
    geometry_msgs::Pose target_pose_;
    geometry_msgs::Pose start_pose_;
 
    moveit::planning_interface::MoveGroupInterface group;
    moveit_msgs::DisplayTrajectory display_trajectory;
    moveit::planning_interface::MoveGroupInterface::Plan my_plan;

    std::vector<double> group_joint_values_;
    std::vector<double> home_joint_values_;
    std::vector<double> start_joint_values_;

    std::vector<geometry_msgs::PoseStamped> goal_pose_list_;
    std::vector<geometry_msgs::PoseStamped>::iterator goal_list_it_;

    robot_state::RobotState *robot_state_;
    robot_state::RobotState *start_state_;

    int N_Joints = 7;
    int plan_seq_ = 0; 

    void initialize_publishers()
    {
        display_publisher = nh_.advertise<moveit_msgs::DisplayTrajectory>("/move_group/display_planned_path", 1, true);
        plan_publisher = nh_.advertise<moveit_msgs::RobotTrajectory>("/planned_path", 1, true);
    }

    public:

    ros::Publisher display_publisher;
    ros::Publisher plan_publisher; 

    trajectory_simulator(): group("arm")
    {
        ROS_INFO("Initializing publisher!!");
        initialize_publishers();
        home_joint_values_.reserve(N_Joints);
        robot_state_ = new robot_state::RobotState(group.getRobotModel());
        start_state_ = new robot_state::RobotState(group.getRobotModel());

        //create home default pose
        //double defaultHome[] = {4.71,2.84,0.00,0.75,4.62,4.48,4.88};
        //double defaultHome[] = {-0.0004588318390768009, 2.8971940876882964, -9.007232158726985, 1.2974520262949687, 16.756792806240767, 7.685565130192222, 32.42986240826462};
        double defaultHome[] = {0.00013466423754771029, 2.9006949211748356, 1.2987090548557987, -2.0797812006459946, -4.881186838651787, -12.566861058232385};
        std::vector<double>::iterator it = home_joint_values_.begin();
        for (int i = 0; i<7; i++, it++)
        {
            (*it) = defaultHome[i];
        }
    };

    ~trajectory_simulator()
    {
        ROS_INFO("Bye!!");        // ^^^^^^^^^^^^^^^^^^^^^^^
        // We can plan a motion for this group to a desired pose for the 
        // end-effector.
    };

    /*Read Configuration File(s)*/
    void configure()
    {

    };

    
    // void setHome(tf::Vector3 _origin, tf::Quaternion _Q)
    // {
    //     Home_.setOrigin(_origin);
    //     Home_.setRotation(_Q);
    //     ROS_INFO("Home set!!");
    //     /* Sleep to give Rviz time to visualize the plan. */
    //     sleep(5.0);
    // };

    void SetStartStateAsHome()
    {
        // get currento state

        //ROS_INFO("Current start state:");      
        //group.setStartStateToCurrentState();
        //group.setStartState(*start_state_);
        //start_state_->printStatePositions();

        //// move to start state = Home (visualization)
        //tf::poseTFToMsg(Home_, target_pose_);
        //ROS_INFO("target pose = Home");
        //group.setPoseTarget(target_pose_);
        //bool success = (group.plan(my_plan) == moveit_msgs::MoveItErrorCodes::SUCCESS);
        //ROS_INFO("Visualizing plan 2 Home (pose goal) %s",success?"":"FAILED");   
        /* Sleep to give Rviz time to visualize the plan. */
        //sleep(5.0);
        //if (success)
        //    group.move();
        //sleep(5.0);

        // Now set start pose to home
        start_state_->setJointGroupPositions(group.getName(),home_joint_values_);
        group.setStartState(*start_state_);
        ROS_WARN("start_state after set: ");
        start_state_->printStatePositions();
    };

    void SetStartState(tf::Vector3 _origin, tf::Quaternion _Q)
    {
        // Given a pose, find joint state values
        Start_.setOrigin(_origin);
        Start_.setRotation(_Q);
        ROS_INFO("Start set!!");
        tf::poseTFToMsg(Start_, start_pose_);
        
        // Find start state pose with the joint state values
        start_state_->setFromIK(group.getRobotModel()->getJointModelGroup(group.getName()), start_pose_);

        // Set the group start state as start_state)_
        group.setStartState(*start_state_);
        ROS_WARN("start_state after set: ");
        start_state_->printStatePositions();
    }

    void setGoal(tf::Vector3 _origin, tf::Quaternion _Q)
    {
        Goal_.setOrigin(_origin);
        Goal_.setRotation(_Q);
        ROS_INFO("Goal set!!");
    };

    void PushGoal(tf::Vector3 _origin, tf::Quaternion _Q)
    {
        // Given a pose, find joint state values
        Goal_.setOrigin(_origin);
        Goal_.setRotation(_Q);
        ROS_INFO("new goal set!!");
        tf::poseTFToMsg(Goal_, target_pose_);

        geometry_msgs::PoseStamped tmp_pose; 
        tmp_pose.pose = target_pose_; 

        goal_pose_list_.push_back(tmp_pose); 

    };

    // void setGoalAsHome()
    // {
    //     Goal_.setOrigin(Home_.getOrigin());
    //     Goal_.setRotation(Home_.getRotation());
    //     ROS_INFO("Goal set = Home!!");
    // };

    /*Free trajectory planning*/
    bool PlanFreeMotion()
    {
        ROS_INFO("Planning Motion!");
        // Planning to a Pose goal
        // ^^^^^^^^^^^^^^^^^^^^^^^
        // We can plan a motion for this group to a desired pose for the 
        // end-effector.
        tf::poseTFToMsg(Goal_, target_pose_);
        ROS_INFO("target pose");
        group.setPoseTarget(target_pose_);
        geometry_msgs::PoseStamped tp = group.getPoseTarget();
        ROS_INFO("target pose set to: x=%f \n y=%f\n z=%f\n xr=%f\n yr=%f\n zr=%f\n w=%f\n ", 
        tp.pose.position.x, tp.pose.position.y, tp.pose.position.z, 
        tp.pose.orientation.x, tp.pose.orientation.y, tp.pose.orientation.z, tp.pose.orientation.w);

        // Now, we call the planner to compute the plan
        // and visualize it.
        // Note that we are just planning, not asking move_group 
        // to actually move the robot.

        //bool success = errorcode.val == 1;
        bool success = (group.plan(my_plan) == moveit_msgs::MoveItErrorCodes::SUCCESS);

        ROS_INFO("Visualizing plan 1 (pose goal) %s",success?"":"FAILED");    
        /* Sleep to give Rviz time to visualize the plan. */
        sleep(5.0);
        return success;

        // TODO
        //if success update start state to funal current state.
    };

    bool PlanMultiStageFreeMotion()
    {
        ROS_INFO("Planning Multi Stage Motion!");
        // Planning to a Pose goal
        // ^^^^^^^^^^^^^^^^^^^^^^^
        // We can plan a motion for this group to a desired pose for the 
        // end-effector.
        group.setPoseTargets(goal_pose_list_);

        std::vector<geometry_msgs::PoseStamped> current_target_list = group.getPoseTargets();
        for (goal_list_it_=current_target_list.begin(); goal_list_it_ < current_target_list.end(); ++goal_list_it_)
        {
            ROS_INFO("target pose set to: x=%1.3f \t y=%1.3f\t z=%1.3f\t xr=%1.3f\t yr=%1.3f\t zr=%1.3f\t w=%1.3f\t ", 
            goal_list_it_->pose.position.x, goal_list_it_->pose.position.y, goal_list_it_->pose.position.z, 
            goal_list_it_->pose.orientation.x, goal_list_it_->pose.orientation.y, goal_list_it_->pose.orientation.z, goal_list_it_->pose.orientation.w);
        }        

        // Now, we call the planner to compute the plan
        // and visualize it.
        // Note that we are just planning, not asking move_group 
        // to actually move the robot.

        //bool success = errorcode.val == 1;
        bool success = (group.plan(my_plan) == moveit_msgs::MoveItErrorCodes::SUCCESS);
        plan_seq_ = ++plan_seq_; 
        my_plan.trajectory_.joint_trajectory.header.seq = plan_seq_; 
        if (success)
            plan_publisher.publish(my_plan.trajectory_);

        ROS_INFO("Visualizing plan 1 (pose goal) %s",success?"":"FAILED");    
        /* Sleep to give Rviz time to visualize the plan. */
        sleep(5.0);

        return success;
    };

    void VisualizePlan()
    {
        ROS_INFO("Visualizing plan");    
        display_trajectory.trajectory_start = my_plan.start_state_;
        display_trajectory.trajectory.push_back(my_plan.trajectory_);
        display_publisher.publish(display_trajectory);
        /* Sleep to give Rviz time to visualize the plan. */
        sleep(5.0);

    };

    void MoveRobot2Goal()
    {
        // Moving to a pose goal
        // ^^^^^^^^^^^^^^^^^^^^^
        //
        // Moving to a pose goal is similar to the step above
        // except we now use the move() function. Note that
        // the pose goal we had set earlier is still active 
        // and so the robot will try to move to that goal. We will
        // not use that function in this tutorial since it is 
        // a blocking function and requires a controller to be active
        // and report success on execution of a trajectory.

        group.move();
        sleep(5.0);
    };

    void ClearGoals()
    {
        goal_pose_list_.clear(); 
    }; 


    /**/
    /**/
};

}




using namespace JacoBoMI;

int main(int argc, char **argv)
{
    ros::init(argc, argv, "trajectory_simulator");
    ros::AsyncSpinner spinner(1);
    spinner.start();

    /* This sleep is ONLY to allow Rviz to come up */
    //sleep(10.0);

    trajectory_simulator tgsim;
    tgsim.configure();

    tf::Vector3 hposit(1.63771402836,1.11316478252, 0.134094119072);
    tf::Quaternion horient (kinova::EulerXYZ2Quaternion(0.212322831154, -0.257197618484, 0.509646713734));
    
    //tgsim.setHome(hposit,horient);
    //tgsim.HomeRobot(); // set start state
    
    // Set the start state
    //tgsim.setStartState(hposit,horient);
    tgsim.SetStartStateAsHome();

    

    tf::Vector3 gposit(0.54882, -0.30854,  0.65841);
    tf::Quaternion gorient(0.68463, -0.22436, 0.68808, 0.086576);
    tgsim.PushGoal(gposit, gorient); 
    // tgsim.setGoal(gposit,gorient);
    // tgsim.PlanFreeMotion(); 

    tf::Vector3 gposit2(0.54882, -0.30854,  0.65841);
    tf::Quaternion gorient2(0.68463, -0.22436, 0.68808, 0.1);
    tgsim.PushGoal(gposit2, gorient2); 

    tf::Vector3 gposit3(0.54882, -0.30854,  0.7);
    tf::Quaternion gorient3(0.68463, -0.22436, 0.68808, 0.1);
    tgsim.PushGoal(gposit3, gorient3); 


    ROS_INFO("Starting multi-stage: ");
    
    tgsim.PlanMultiStageFreeMotion(); 
    //tgsim.MoveRobot2Goal();

    //tgsim.setGoalAsHome();
    //tgsim.PlanFreeMotion(); 
    //tgsim.MoveRobot2Goal();

    sleep(5.0);
    tgsim.VisualizePlan();

    tgsim.ClearGoals(); 


    // std_msgs::String a_string;
    // a_string.data = "hi";

    // while (ros::ok())
    // {
    //     tgsim.display_publisher.publish(a_string); 
    //     ros::spinOnce();
    //     sleep(10.0);
    // }

    int a;
    std::cin >> a;

    ROS_INFO("Bye!!");
    ros::shutdown();

  return 0;
}
