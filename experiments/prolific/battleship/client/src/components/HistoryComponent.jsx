import React, { useEffect, useState } from 'react';
import { usePlayer, useRound, useStage } from "@empirica/core/player/classic/react";
import { noQuestion, noQuestionAnswer, timeoutAnswer } from '../../../utils/systemText.js';

export function HistoryComponent(grid) {
  const round = useRound();
  const stage = useStage();
  const player = usePlayer();

  var msgHistory = round.get("messages");
  
  if (stage.get("name") == "instructions") {
    msgHistory = player.stage.get("messages");
  }

  const pastMessages = msgHistory.slice();

  function capitalize(s) {
    return s.charAt(0).toUpperCase() + s.slice(1);
  }

  function getAuthor(s) {
    switch (s) {
      case "question":
        return "Captain";
      case "answer":
        return "Spotter";
      case "move":
        return "Move";
    }
  }

  function handleText(message) {
    var text = message.text;
    if ((text == noQuestion || text == noQuestionAnswer) || text == timeoutAnswer) {
      if (message.type == "question") {
      return <div><b>[Question skipped or timed out]</b></div>
      }
    } else {
      return <div style={{marginBottom: message.type == "move" ? "20px" : "0px"}}><b>{capitalize(getAuthor(message.type))}</b>: <i>{capitalize(message.text)}</i></div>
    }
  }

  var histHeight = 0;

  if (!(stage.get("name") == "instructions")) {
    var histHeight = "20vw";
  } else {
    var histHeight = "15vw";
  }

  return (
     <div style={{margin: '20px', padding:'15px', boxShadow:"3px 3px 3px 3px rgba(0,0,0,0.1)", display:"flex", flexDirection:"column", alignItems:"center"}}>
        <u style={{fontSize: "1.25vw"}}>Chat History</u>
        <div style={{ width: '25vw', height: histHeight, overflowY: 'scroll', backgroundColor: 'white', color: 'black', display: 'flex', flexDirection: 'column-reverse', border:"rgba(0,0,0,0.1)" }}>
      {pastMessages.toReversed().map((message) => (
        <div style={{fontSize: "1.1vw"}}>
          {handleText(message)}
        </div>
      ))}
    </div>
     </div>

  );
}

export default HistoryComponent;
