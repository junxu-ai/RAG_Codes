from confluent_kafka import Consumer, KafkaException
import logging
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def consume_streaming_data():
    consumer_conf = {
        'bootstrap.servers': 'localhost:9092',
        'group.id': 'my_consumer_group',
        'auto.offset.reset': 'earliest',
    }
    consumer = Consumer(consumer_conf)
    topic = 'real_time_data'

    try:
        consumer.subscribe([topic])
        logging.info(f"Subscribed to topic: {topic}")
        
        while True:
            msg = consumer.poll(timeout=1.0)
            if msg is None:
                continue
            elif msg.error():
                raise KafkaException(msg.error())
            else:
                # Process the streaming data
                try:
                    data = msg.value().decode('utf-8')
                    logging.debug(f"Received message: {data}")
                    update_knowledge_base_with_streaming_data(data)
                except Exception as e:
                    logging.error(f"Error processing message: {e}")
                    
    except KafkaException as e:
        logging.error(f"Kafka error: {e}")
    except KeyboardInterrupt:
        logging.info("Consumer interrupted by user")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
    finally:
        consumer.close()
        logging.info("Consumer closed")

def update_knowledge_base_with_streaming_data(data):
    """
    Update knowledge base with streaming data
    Replace with actual implementation
    """
    try:
        # Example implementation:
        # parsed_data = json.loads(data) if needed
        # database.update(parsed_data)
        logging.info(f"Updating knowledge base with data: {data[:100]}...")  # Log first 100 chars
    except Exception as e:
        logging.error(f"Error updating knowledge base: {e}")

if __name__ == "__main__":
    # Start consuming streaming data
    consume_streaming_data()