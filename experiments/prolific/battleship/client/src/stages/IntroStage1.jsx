import React, { useState, useReducer } from "react";
import { usePlayer, useStage, useGame, useRound, useStageTimer } from "@empirica/core/player/classic/react";
import BoardComponent from "../components/BoardComponent.jsx";
import ShipsRemainingComponent from "../components/ShipsRemainingComponent.jsx";
import { Button } from "../components/Button.jsx";
import { SpotterBoardComponent } from "../components/SpotterBoardComponent.jsx";
import { HistoryComponent } from "../components/HistoryComponent.jsx";
import { noQuestion, timeoutAnswer } from "../../../utils/systemText.js";
import { SendMessageIntro } from "../../../utils/SendMessage.js";
import { InstructionBox } from "../components/InstructionBox.jsx";

export function IntroStage1() {
    const player = usePlayer();
    const round = useRound();
    const stage = useStage();
    const game = useGame();
    const timer = useStageTimer();

    const [showTextbox, setShowTextbox] = useState(false);
    const [showBackButton, setShowBackButton] = useState(true);

    if (game.get("questionEveryTime") && showBackButton) {
        setShowTextbox(true);
        setShowBackButton(false);
    }

    function handleAskQuestion() {
        if (!showTextbox) {
            setShowTextbox(true);
        } else {
            askQuestion();
        }
    }

    function answerQuestion(inputType, hardcodedAnswer) {
        if (player.stage.get("question") == noQuestion) {
            var inputText = "";
        } else {
            switch (inputType) {
                case "free":
                    var inputText = document.getElementById('answer').value;
                    inputText = inputText.trim();
                    break;
                case "categorical":
                    var inputText = hardcodedAnswer;
            }
        }

        if (inputText.length > 1 && inputText.length < 20) {
            SendMessageIntro(inputText, "answer", player);
            player.stage.set("textStage",player.stage.get("textStage")+1)
        } 
        if (inputText.length == 0) {
            if (player.stage.get("question") == noQuestion) {
                inputText = noQuestionAnswer;
            } else {
                inputText = timeoutAnswer;
            }
            SendMessageIntro(inputText, "answer", player);
            player.stage.set("textStage",player.stage.get("textStage")+1)
        }
    }


    function getPossibleAnswers() {
        if (game.get("categoricalAnswers")) {
            return (
            <div style={{display: "flex", flexDirection: "row", alignItems: "center"}}>
                <Button className="m-5" primary={true} handleClick={() => alert("[Tutorial] The red ship is vertical: please answer 'No'.")}>Yes</Button>
                <Button className="m-5" handleClick={() => answerQuestion("categorical", "no")}>No</Button>
            </div>
            );
        } else {
            return (
            <div style={{display: "flex", flexDirection: "row", alignItems: "center"}}>
                <input type="text" id="answer" name="answer" style={{height: "50px"}} autoComplete="off"/>
                <Button className="m-5" handleClick={() => answerQuestion("free")}>Answer Question</Button>
            </div>);
        }
    }
    
    function askQuestion() {
        var inputText = document.getElementById('question').value;
        inputText = inputText.trim();
        if (inputText == "Is the red ship horizontal?" || inputText == "is the red ship horizontal?" || inputText == "is the red ship horizontal" || inputText == "Is the red ship horizontal") {
            SendMessageIntro(inputText, "question", player);
            SendMessageIntro("No", "answer", player);
            player.stage.set("textStage",player.stage.get("textStage")+1);
        } else {
            alert("[Tutorial] Please ask 'Is the red ship horizontal?'");
        }
    }

    if (player.stage.get("textStage") == 7 && player.stage.get("forcedUpdate") == undefined) {
        window.location.reload(false);
        player.stage.set("forcedUpdate",true);
    }

  return (<div className="mt-3 sm:mt-5" style={{display:"flex", flexDirection:"column", alignItems:"center"}}>
      <div style={{display: "flex", flexDirection: "column", alignItems: "center"}}> 
           <InstructionBox textStage={player.stage.get("textStage")}/>
          <div style={{display: "flex", flexDirection: "row", alignItems: "center"}}>

            {player.stage.get("textStage") > 2 ?
              <div style={{display: "flex", flexDirection: "column", margin: "30px", alignItems: "center", paddingTop:"50px"}}>
                  <ShipsRemainingComponent 
                      shipsStatus={player.stage.get("shipStatus")}
                  />
              </div> : <div></div>
            }


            {player.stage.get("textStage") > 7 
            ? <SpotterBoardComponent
            occ_tiles={player.stage.get("occTiles")}
            init_tiles={player.stage.get("trueTiles")}
            ships={player.stage.get("ships")}
                 />
            :  <BoardComponent 
                  init_tiles={player.stage.get("occTiles")}
                  ships={player.stage.get("ships")}
              />
            }

            {player.stage.get("textStage") > 5 ?
              <HistoryComponent
                  externalMsg={player.stage.get("messages")}
              /> : <div></div>
            }
          </div>
          {player.stage.get("textStage") > 9 ? <div></div>
          : player.stage.get("textStage") == 9 
          ? <div style={{display:"flex", flexDirection:"column", alignItems:"center"}}><p style={{margin: "10px", fontSize:"1.2vw"}}>The captain asked the following question: <i>"{player.stage.get("question")}"</i>. Please answer it below:</p>
            {getPossibleAnswers()}</div>
          : player.stage.get("textStage") > 6
          ? <div></div>
          : (player.stage.get("textStage") == 6) 
          ? <div style={{display:"flex", flexDirection:"column", textAlign:"center", fontSize:"1.2vw"}}>
            <p>You asked <b>"{player.stage.get('question')}"</b>. The spotter answered <b>"{player.stage.get('answer')}"</b>.</p>
            <p>Please click a tile to fire at it!</p>
            </div>            
          : (player.stage.get("textStage") < 5)
          ? <div></div>
          : <div style={{display:"flex", flexDirection:"column", alignItems:"center"}}> 
            <div className="flex justify-center" style={{display: "flex", flexDirection: "row", alignItems: "center"}}>
                    {showTextbox && (
                <div style={{display:"flex", flexDirection:"row", alignItems: "center"}}>
                <input type="text" id="question" name="question" style={{height: "50px"}} autoComplete="off"/>
                        <Button className="m-5" handleClick={handleAskQuestion}>
                        {showTextbox ? "Submit Question" : "Ask Question"}
                        </Button>
                </div>
        )     }
          { !showTextbox && (
              <div style={{display:"flex", flexDirection:"row", alignItems: "center"}}><Button className="m-5" handleClick={handleAskQuestion}>
                      Ask Question
                   </Button>
              <p style={{fontSize: "20px"}}>or</p>
                  <Button className="m-5" primary={true} handleClick={() => alert("[Tutorial] Just for the tutorial, please ask a question :)")}>Ready to Fire!</Button>
                  </div>)
          }
          { (showTextbox && showBackButton) 
          ? (<div> <Button className="m-5" handleClick={() => setShowTextbox(false)}>Back</Button></div>)
          : <div></div>
          }
          </div>
          <div style={{display:"flex", flexDirection:"column", alignItems:"center"}}>
          {game.get("categoricalAnswers") ? <p style={{margin: "0px", fontSize: "12px"}}>Remember: the spotter can only answer your questions with "yes" or "no"!</p> : <div></div>}
          {showBackButton ? <p style={{margin: "10px", fontSize: "17px"}}><i>You can ask <b>{player.stage.get("questionsRemaining")}</b> more questions.</i></p> : <div></div>}
          </div>
      </div>}

          </div>
  </div>
);
} 
