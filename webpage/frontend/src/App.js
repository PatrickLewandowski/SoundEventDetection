import React, { useState, useEffect} from "react";
import "./App.css";
import LineChart from "./LineChart";
import YouTube from 'react-youtube';
import TextField from '@material-ui/core/TextField';
import Button from '@material-ui/core/Button';
import IconButton from '@material-ui/core/IconButton';
import InputBase from '@material-ui/core/InputBase';
import { makeStyles } from '@material-ui/core/styles';
import Icon from '@material-ui/core/Icon';
// import ProductDetail from "./data";
import { w3cwebsocket as W3CWebSocket } from "websocket";


const useStyles = makeStyles(theme => ({
  root: {
    padding: '2px 4px',
    display: 'flex',
    alignItems: 'center',
    width: 500,
    marginLeft: 'auto',
    marginRight: 'auto',
  },
  input: {
    // marginLeft: theme.spacing(1),
    marginRight: theme.spacing(1),
    marginLeft: 'auto',
    // marginRight: 'auto',
    alignItems: 'center',
    flex: 1,
  },
  iconButton: {
    marginLeft: 10,
    padding: 10,
  },
  form: {
    width: '100%', // Fix IE 11 issue.
    marginTop: theme.spacing(1),
  },
  submit: {
    margin: theme.spacing(3, 0, 2),
  },
}));

const opts = {
      height: '390',
      width: '640',
      playerVars: {
        autoplay: 1
      }
    };

var videoTarget = null;

function App() {

  const[lineData, setLineData] = useState([]);
  const [videoID,setVideoID] = useState('');
  const [videoProgress, setVideoProgess] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isReady, setIsReady] = useState(false);
  const [link,setLink] = useState("")
  const [buttonDisabled, setButtonDisabled] = useState(true)

  function stateChange(event) {
    console.log(event.data)
    switch (event.data) {
      case YouTube.PlayerState.PAUSED:
      setIsPlaying(false);
      break;

      case YouTube.PlayerState.ENDED:
      setIsPlaying(false);
      break;

      case YouTube.PlayerState.PLAYING:
      setIsPlaying(true);
      break;
    }
  }
  function onReady(event) {
    setIsReady(true);
    event.target.pauseVideo();
    videoTarget = event.target;
  }

  const classes = useStyles();


  function  handleChange (e) {
      setLink(e.target.value);
      console.log(e.target.value);
      setButtonDisabled(e.target.value.length == 0);
    }

    
  function handleClick (e) {

    setLineData([]);

    var video_id = link.split('v=')[1];
    console.log(video_id);
    setVideoID(video_id);

    const dataClient = new W3CWebSocket('ws://localhost:8000/');

    dataClient.onopen = () => {
      function sendVideoLink() {
        if (dataClient.readyState === dataClient.OPEN) {
          dataClient.send(link);
          setTimeout(sendVideoLink, 1000);
        }
      }
      sendVideoLink();
    }
      
      dataClient.onmessage = (msg) => {
        var response = JSON.parse(msg.data);
        var head = response["event"];
        var head_body = response[1];
        if ( head != "subscribed" ) {
          setLineData(response);
          setIsPlaying(true);
        }
      }
    }

    useEffect(() => {
      let interval = null;

      if (isReady && isPlaying) {
      videoTarget.playVideo();
      interval = setInterval(function() {setVideoProgess(videoProgress => videoProgress + 1);},1000);
    } else {
      clearInterval(interval);
    }

    return () => clearInterval(interval);

  },[isPlaying,isReady, videoProgress]);
  

    return (

    <React.Fragment>

      <div>
      <form className={classes.root} noValidate autoComplete="off">
      <InputBase
        className={classes.input}
        placeholder="Youtube Link"
        inputProps={{ 'aria-label': 'youtube link' }}
        onChange={(e) => {handleChange(e);}}
        onKeyPress={(e) => {
            if (e.key === 'Enter') {
              e.preventDefault();
              handleClick(e);

            }
    }}
      />

      <Button
        variant="contained"
        color="primary"
        className={classes.button}
        endIcon={<Icon>send</Icon>}
        disabled={buttonDisabled}
        onClick={(e) => { handleClick(e);}}
      >
        Detect
      
      </Button> 
      </form>


      </div>

      { videoID.length > 0 &&
        <div>
      <YouTube
        videoId={videoID}
        opts={opts}
        onReady={(e) => onReady(e)}
        onStateChange={(e) => stateChange(e)}
      />
      
      </div>
      }
      
      {lineData.length > 0 &&
        <div>
        <h2>Probabilities</h2>
        <LineChart data={lineData} progressLine={videoProgress} />
        </div>
      }

    </React.Fragment>
  );
}

export default App;
