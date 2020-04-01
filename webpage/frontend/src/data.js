import React from 'react';
import Websocket from 'react-websocket';
import { w3cwebsocket as W3CWebSocket } from "websocket";


 
class ProductDetail extends React.Component {
 
    constructor(props) {
      super(props);
      this.state = {
        count: 92
      };

      const client = new W3CWebSocket('ws://localhost:8000/');

      client.onopen = () => {
        function sendNumber() {
        if (client.readyState === client.OPEN) {
            var number = Math.round(Math.random() * 0xFFFFFF);
            client.send(number.toString());
            setTimeout(sendNumber, 1000);
        }
    }
    sendNumber();
      }

      client.onmessage = (msg) => {
      var response = JSON.parse(msg.data);
      var head = response["event"];
      var head_body = response[1];
      if ( head == "subscribed" ) {
        console.log("channelID = ", response["chanId"], " currency = ", response["pair"]);
      } else {
        console.log(msg.data);
        let result = JSON.parse(msg.data);
      console.log(result)
      this.setState({count: this.state.count + result.count});
      }
    }

    }

    

 
    handleData(data) {
      console.log(data)
      let result = JSON.parse(data);
      console.log(result)
      this.setState({count: this.state.count + result.count});
    }
 
    render() {
      return (
        <div>
          Count: <strong>{this.state.count}</strong>
        </div>
      );
    }
  }
 
  export default ProductDetail;