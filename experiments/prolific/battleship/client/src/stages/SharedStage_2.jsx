import React, { useState } from "react";
import { usePlayer, useStage, useGame, useRound, useStageTimer } from "@empirica/core/player/classic/react";
import BoardComponent from "../components/BoardComponent.jsx";
import ShipsRemainingComponent from "../components/ShipsRemainingComponent.jsx";
import { Button } from "../components/Button.jsx";
import { SpotterBoardComponent } from "../components/SpotterBoardComponent.jsx";
import { noQuestion, noQuestionAnswer, timeoutAnswer } from "../../../utils/systemText.js";
import { SendMessage } from "../../../utils/SendMessage.js";
import { HistoryComponent } from "../components/HistoryComponent.jsx";

export function SharedStage_2() {

    const game = useGame();
    const stage = useStage();
    const timer = useStageTimer();
    const round = useRound();
    const player = usePlayer();
    const [questionRating, setQuestionRating] = useState(3);

    function handleQuestionRatingChange(e) {
        setQuestionRating(e.target.value);
    }

    function getAdditionalText(index) {
        switch (index) {
            case 0:
                return "(Not at all helpful)";
            case 1:
                return "(Slightly helpful)";
            case 2:
                return "(Somewhat helpful)";
            case 3:
                return "(Very helpful)";
            case 4:
                return "(Extremely helpful)";
          default:
            return ""
        }
    }
    function getPossibleAnswers() {
        if (game.get("categoricalAnswers")) {
            return (
            <div style={{display: "flex", flexDirection: "row", alignItems: "center"}}>
                <Button className="m-5" handleClick={() => answerQuestion("categorical", "yes")}>Yes</Button>
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

    function handleSpotterSubmission(){
        round.set("spotterRatings",[...round.get("spotterRatings"), [round.get("question"), questionRating]]);
        stage.set("questionRated",true);
    }
    
    function handleSpotterLikert(){
        var lastQuestion = round.get("question");
        console.log("last q", lastQuestion);
            if (lastQuestion != noQuestion && lastQuestion != undefined) {
                if (!stage.get("questionRated")) {
                return (<div style={{display:"flex", flexDirection:"row", justifyItems:"center"}}><div>
                    <label className={"block text-sm font-medium text-gray-700 my-2"}>
                    <p style={{fontSize:"1.25vw", marginRight:"0.5vw"}}> <b>How helpful is this question?:</b></p>
                    </label>
                    <div className="flex space-x-4" style={{justifyContent:"center"}}>
                      {[...Array(5)].map((_, index) => (
                    <label key={index} className="flex items-center space-x-1">
                      <input
                        type="radio"
                        name="questionRating"
                        value={index + 1}
                        checked={questionRating === (index + 1).toString()}
                        onChange={handleQuestionRatingChange}
                      />
                      {index + 1} {getAdditionalText(index)}
                    </label>
                  ))}
                    </div>
                  </div>
                    <div style={{margin:"1vw"}}>
                    <Button handleClick={handleSpotterSubmission} width="8vw" height="6vh">Rate</Button>
                    </div>    
                </div>);
                }
                else {
                    return (<div><i>Thank you for rating. The captain is thinking of a question...</i></div>);
                }
            } else {
                player.stage.set("submit",true);
                return (<div style={{margin: "20px", fontSize:"1vw"}}><i>The captain is thinking of a question...</i></div>);
            }
    }

    function answerQuestion(inputType, hardcodedAnswer) {
        if (round.get("question") == noQuestion) {
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
            SendMessage(inputText, "answer", round, game, timer);
            stage.set("answered",true);
        } 
        if (inputText.length == 0) {
            if (round.get("question") == noQuestion) {
                inputText = noQuestionAnswer;
            } else {
                inputText = timeoutAnswer;
            }
            SendMessage(inputText, "answer", round, game, timer);
            stage.set("answered",true);
        }
    }

    if (round.get("question") == undefined) {
        round.set("question", noQuestion);
    }

    switch (player.round.get("role")) {
        case "spotter":
        if ((stage.get("questionRated") && stage.get("answered")) || (stage.get("answered") && !game.get("spotterRatesQuestions"))) {
            player.stage.set("submit",true);
        }
        if (!game.get("spotterRatesQuestions")) {
            if (round.get("question") != noQuestion) {
                return (<div style={{display: "flex", flexDirection: "column", alignItems: "center"}}>
                
                    <div style={{display: "flex", flexDirection: "row", alignItems: "center"}}>
                        <div style={{display: "flex", flexDirection: "column", margin: "30px", alignItems: "center", paddingTop:"50px"}}>
                        <ShipsRemainingComponent 
                            shipsStatus={round.get("shipsSunk")}
                        />
                        </div>
                        <SpotterBoardComponent 
                        occ_tiles={round.get("occTiles")}
                            init_tiles={round.get("trueTiles")}
                            ships={round.get("ships")}
                        />
                    <HistoryComponent
                        grid = {round.get("occTiles")[0].length}/>
                    </div>
                    <p style={{margin: "20px"}}>The captain asked the following question: <i>"{round.get("question")}"</i>. Please answer it below:</p>
                    {getPossibleAnswers()}
                    </div>);
              } else {
                answerQuestion();
              }
        }
        else {
          if (round.get("question") != noQuestion) {
            return (<div style={{display: "flex", flexDirection: "column", alignItems: "center"}}>
            
                <div style={{display: "flex", flexDirection: "row", alignItems: "center"}}>
                    <div style={{display: "flex", flexDirection: "column", margin: "30px", alignItems: "center", paddingTop:"50px"}}>
                    <ShipsRemainingComponent 
                        shipsStatus={round.get("shipsSunk")}
                    />
                    </div>
                    <SpotterBoardComponent 
                    occ_tiles={round.get("occTiles")}
                        init_tiles={round.get("trueTiles")}
                        ships={round.get("ships")}
                    />
                <HistoryComponent
                    grid = {round.get("occTiles")[0].length}/>
                </div>
                <div style={{display: "flex", flexDirection: "column", alignItems: "center"}}>
                <p style={{margin: "20px"}}>The captain asked the following question: <i>"{round.get("question")}"</i>. Please answer it below:</p>
                {!stage.get("answered") ? getPossibleAnswers() : handleSpotterLikert()}
                </div>

                </div>);
          } else {
            answerQuestion();
          }
        }
        case "captain":
          player.stage.set("submit", true);
          return (<div style={{display: "flex", flexDirection: "column", alignItems: "center"}}> 
            <div style={{display: "flex", flexDirection: "row", alignItems: "center"}}>
                <div style={{display: "flex", flexDirection: "column", margin: "30px", alignItems: "center", paddingTop:"50px"}}>
                <ShipsRemainingComponent 
                    shipsStatus={round.get("shipsSunk")}
                />
                </div>
                <BoardComponent 
                    init_tiles={round.get("occTiles")}
                    ships={round.get("ships")}
                />
                <HistoryComponent
                    grid = {round.get("occTiles")[0].length}/>
            </div>
            <p style={{margin: "20px"}}><i>The spotter is answering your question...</i> </p>
            </div>);
        default:
          return <div>This is the shared stage. You have no role: something's gone wrong!</div>;
      }
}