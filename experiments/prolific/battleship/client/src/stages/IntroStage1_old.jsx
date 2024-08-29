import React, { useState } from "react";
import { usePlayer, useStage, useGame, useRound, useStageTimer } from "@empirica/core/player/classic/react";
import BoardComponent from "../components/BoardComponent.jsx";
import ShipsRemainingComponent from "../components/ShipsRemainingComponent.jsx";
import { Button } from "../components/Button.jsx";
import { SpotterBoardComponent } from "../components/SpotterBoardComponent.jsx";
import { HistoryComponent } from "../components/HistoryComponent.jsx";
import { noQuestion, timeoutAnswer } from "../../../utils/systemText.js";
import { SendMessageIntro } from "../../../utils/SendMessage.js";

export function IntroStage1() {
    const player = usePlayer();
    const round = useRound();
    const stage = useStage();
    const game = useGame();
    const timer = useStageTimer();

    const [showTextbox, setShowTextbox] = useState(false);
    const [hasFired, setHasFired] = useState(false);

    function handleAskQuestion() {
        if (!showTextbox) {
            setShowTextbox(true);
        } else {
            askQuestion();
        }
    }
    
    function askQuestion() {
        var inputText = document.getElementById('question').value;
        inputText = inputText.trim();
        if ((inputText.length >= 1 && inputText.length < 100) && inputText == "Is the red ship horizontal?") {
            SendMessageIntro(inputText, "question", player);
            SendMessageIntro("No", "answer", player);
            player.stage.set("introStage",3);
        } else {
            alert("[Tutorial] Please ask 'Is the red ship horizontal?'");
        }
    }

  return (<div className="mt-3 sm:mt-5" style={{display:"flex", flexDirection:"column", alignItems:"center"}}>
      <div style={{display: "flex", flexDirection: "column", alignItems: "center"}}> 
          <div style={{display: "flex", flexDirection: "row", alignItems: "center"}}>
              <div style={{display: "flex", flexDirection: "column", margin: "30px", alignItems: "center", paddingTop:"50px"}}>
                  <ShipsRemainingComponent 
                      shipsStatus={player.stage.get("shipStatus")}
                  />
              </div>
              <BoardComponent 
                  init_tiles={player.stage.get("occTiles")}
                  ships={player.stage.get("ships")}
              />
              <HistoryComponent
                  externalMsg={player.stage.get("messages")}
              />
          </div>
          <p style={{margin: "20px", marginBottom: "0px"}}>You are the <b>captain</b>. Your goal is to sink all the hidden ships. You can ask the spotter questions to get information about the board and inform where you should shoot next.</p>
          <p style={{margin: "20px", marginTop: "10px", fontSize:"1.2vw", marginBottom:"0px"}}> <i>For this tutorial, try asking: "<b>Is the red ship horizontal?</b>"</i></p>
          <div className="flex justify-center" style={{display: "flex", flexDirection: "row", alignItems: "center"}}>
              {showTextbox && (
                  <div style={{display:"flex", flexDirection:"row", alignItems: "center"}}>
                  <input type="text" id="question" name="question" style={{height: "50px"}} autocomplete="off"/>
                        <Button className="m-5" handleClick={handleAskQuestion}>
                        {showTextbox ? "Submit Question" : "Ask Question"}
                        </Button>
                  </div>
              )}
              { !showTextbox && (
                  <div style={{display:"flex", flexDirection:"row", alignItems: "center"}}><Button className="m-5" handleClick={handleAskQuestion}>
                          Ask Question
                       </Button>
                  <p style={{fontSize: "20px"}}>or</p>
                      <Button className="m-5" primary={true} handleClick={() => alert("[Tutorial] Just for the tutorial, please ask a question :)")}>Ready to Fire!</Button></div>)
              }
              { showTextbox && (
                  <div> <Button className="m-5" handleClick={() => setShowTextbox(false)}>Back</Button></div>)
              }
          </div>
          <p style={{margin: "0px", fontSize: "12px"}}>Remember: the spotter can only answer your questions with "yes" or "no"!</p>
          <p style={{margin: "10px", fontSize: "17px"}}><i>You can ask <b>{player.stage.get("questionsRemaining")}</b> more questions.</i></p>
          </div>
  </div>
);
} 
