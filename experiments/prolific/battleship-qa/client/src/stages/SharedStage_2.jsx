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
    const timer = useStageTimer();
    const round = useRound();
    const player = usePlayer();

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
            player.stage.set("submit",true);
        } 
        if (inputText.length == 0) {
            if (round.get("question") == noQuestion) {
                inputText = noQuestionAnswer;
            } else {
                inputText = timeoutAnswer;
            }
            SendMessage(inputText, "answer", round, game, timer);
            player.stage.set("submit",true);
        }
    }

    if (round.get("question") == undefined) {
        round.set("question", noQuestion);
    }

    switch (player.round.get("role")) {
        case "spotter":
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