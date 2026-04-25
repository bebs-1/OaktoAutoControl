#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/int32_multi_array.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "my_robot_pkg/SparkMax.hpp"
#include <chrono>
#include <iomanip>
#include <memory>
#include <iostream>
#include <string>


class MotorDriver : public rclcpp::Node
{
public:
    MotorDriver() : Node("motor_driver"), motor1("can0", 1), motor2("can0", 2), 
                    motor3("can0", 3), motor4("can0", 4)
    {
        // Configure motors
        motor1.SetMotorType(MotorType::kBrushless);
        motor2.SetMotorType(MotorType::kBrushless);
        motor3.SetMotorType(MotorType::kBrushless);
        motor4.SetMotorType(MotorType::kBrushless);

        
        
        motor1.SetIdleMode(IdleMode::kBrake);
        motor2.SetIdleMode(IdleMode::kBrake);
        motor3.SetIdleMode(IdleMode::kBrake);
        motor4.SetIdleMode(IdleMode::kBrake);
        
        motor1.SetSensorType(SensorType::kHallSensor);
        motor2.SetSensorType(SensorType::kHallSensor);
        motor3.SetSensorType(SensorType::kHallSensor);
        motor4.SetSensorType(SensorType::kHallSensor);

        motor1.SetP(0, 0.0002f);
        motor1.SetI(0, 0.0f);
        motor1.SetD(0, 0.0f);
        motor1.SetF(0, 0.000205f);

        motor2.SetP(0, 0.0002f);
        motor2.SetI(0, 0.0f);
        motor2.SetD(0, 0.0f);
        motor2.SetF(0, 0.000205f);

        motor3.SetP(0, 0.0002f);
        motor3.SetI(0, 0.0f);
        motor3.SetD(0, 0.0f);
        motor3.SetF(0, 0.000205f);

        motor4.SetP(0, 0.0002f);
        motor4.SetI(0, 0.0f);
        motor4.SetD(0, 0.0f);
        motor4.SetF(0, 0.000205f);

        motor1.BurnFlash();
        motor2.BurnFlash();
        motor3.BurnFlash();
        motor4.BurnFlash();

        // Create subscriber for motor commands
        sub_ = this->create_subscription<std_msgs::msg::Int32MultiArray>(
            "motor_cmd", 10,
            std::bind(&MotorDriver::cmd_callback, this, std::placeholders::_1));

        // Create publisher for motor data
        publisher_ = this->create_publisher<sensor_msgs::msg::JointState>(
            "motor_data", 10);

        // Create timer to publish motor data at 10 Hz
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(100),
            std::bind(&MotorDriver::publish_motor_data, this));
    }

private:
    SparkMax motor1;
    SparkMax motor2;
    SparkMax motor3;
    SparkMax motor4;

    rclcpp::Subscription<std_msgs::msg::Int32MultiArray>::SharedPtr sub_;
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr publisher_;
    rclcpp::TimerBase::SharedPtr timer_;

    void publish_motor_data()
    {
        auto message = sensor_msgs::msg::JointState();
        message.header.stamp = this->now();

        message.name     = {"motor1", "motor2", "motor3", "motor4"};
        message.velocity = {
            static_cast<double>(motor1.GetVelocity()),
            static_cast<double>(motor2.GetVelocity()),
            static_cast<double>(motor3.GetVelocity()),
            static_cast<double>(motor4.GetVelocity())
        };
        message.effort = {
            static_cast<double>(motor1.GetCurrent()),
            static_cast<double>(motor2.GetCurrent()),
            static_cast<double>(motor3.GetCurrent()),
            static_cast<double>(motor4.GetCurrent())
        };

        publisher_->publish(message);
        RCLCPP_DEBUG(this->get_logger(), "Published motor data");
    }

    void cmd_callback(const std_msgs::msg::Int32MultiArray::SharedPtr msg)
    {
        if (msg->data.size() < 4) {
            RCLCPP_WARN(this->get_logger(), "Expected 4 values, got %zu", msg->data.size());
            return;
        }
        SparkBase::Heartbeat();
        motor1.SetVelocity(msg->data[0]);
        motor2.SetVelocity(msg->data[1]);
        motor3.SetVelocity(msg->data[2]);
        motor4.SetVelocity(msg->data[3]);
        RCLCPP_INFO(this->get_logger(), "CMD: [%d, %d, %d, %d]",
            msg->data[0], msg->data[1], msg->data[2], msg->data[3]);
    }
};




int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<MotorDriver>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}

