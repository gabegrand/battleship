import { usePlayer } from "@empirica/core/player/classic/react";
import React, { useState } from "react";
import { Alert } from "../components/Alert";
import { Button } from "../components/Button";

export function ExitSurvey({ next }) {
  const labelClassName = "block text-sm font-medium text-gray-700 my-2";
  const inputClassName =
    "appearance-none block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm placeholder-gray-400 focus:outline-none focus:ring-empirica-500 focus:border-empirica-500 sm:text-sm";
  const player = usePlayer();

  const [partnerLikertValue, setPartnerLikertValue] = useState(4);
  const [studyLikertValue, setStudyLikertValue] = useState(4);
  const [feedback, setFeedback] = useState("");

  function handleSubmit(event) {
    event.preventDefault();
    player.set("exitSurvey", {
      partnerLikertValue,
      studyLikertValue,
      feedback,
    });
    next();
  }

  function handlePartnerLikertChange(e) {
    setPartnerLikertValue(e.target.value);
  }

  function handleStudyLikertChange(e) {
    setStudyLikertValue(e.target.value);
  }

  function getAdditionalText(index) {
    switch (index) {
      case 0:
        return "(Extremely Unlikely)";
      case 1:
        return "(Very Unlikely)";
      case 2:
        return "(Unlikely)";
      case 3:
        return "(Neutral)";
      case 4:
        return "(Likely)";
      case 5:
        return "(Very Likely)";
      case 6:
        return "(Extremely Likely)";
      default:
        return ""
    }
  }

  return (
    <div className="py-8 max-w-5xl mx-auto px-4 sm:px-6 lg:px-8">

      <form
        className="mt-12 space-y-8 divide-y divide-gray-200"
        onSubmit={handleSubmit}
      >
        <div className="space-y-8 divide-y divide-gray-200">
          <div>
            <div>
              <h3 className="text-lg leading-6 font-medium text-gray-900">
                Exit Survey
              </h3>
              <p className="mt-1 text-sm text-gray-500">
                Please answer the following short survey. You do not have to
                provide any information you feel uncomfortable with.
              </p>
              {player.get("stuck") ? <h1 style={{fontSize:"1.5vw", marginLeft:"0vw", marginTop:"1vh"}}>The experiment was ended early by one of the participants.</h1> : <div></div>}
              <h1 style={{fontSize:"2vw", margin:"2vw", marginLeft:"0vw", marginBottom:"1vh"}}>Your Prolific completion code is <b>{player.get("stuck") ? player.get("compcodeStuck") : player.get("compcode")}</b></h1>
              <h1 style={{fontSize:"1.4vw", margin:"2vw", marginLeft:"0vw", marginTop:"1vh"}}>Make sure to paste it into Prolific to ensure payment.</h1>
            </div>

            <div className="space-y-8 mt-6">

              <div>
                <label className={labelClassName}>
                <b>Rate your partner:</b> If you were to play the game again, how likely would you be to pick this partner?
                </label>
                <div className="flex space-x-4">
                  {[...Array(7)].map((_, index) => (
                <label key={index} className="flex items-center space-x-1">
                  <input
                    type="radio"
                    name="partnerLikert"
                    value={index + 1}
                    checked={partnerLikertValue === (index + 1).toString()}
                    onChange={handlePartnerLikertChange}
                  />
                  {index + 1} {getAdditionalText(index)}
                </label>
              ))}
                </div>
              </div>

              <div>
                <label className={labelClassName}>
                <b>Rate the study:</b> How likely would you be to recommend this study to a friend?
                </label>
                <div className="flex space-x-4">
                  {[...Array(7)].map((_, index) => (
                <label key={index} className="flex items-center space-x-1">
                  <input
                    type="radio"
                    name="studyLikert"
                    value={index + 1}
                    checked={studyLikertValue === (index + 1).toString()}
                    onChange={handleStudyLikertChange}
                  />
                  {index + 1} {getAdditionalText(index)}
                </label>
              ))}
                </div>
              </div>

              <div className="grid grid-cols-3 gap-x-6 gap-y-3">

                <label className={labelClassName}>
                  Feedback, including problems you encountered.
                </label>

                <textarea
                  className={inputClassName}
                  dir="auto"
                  id="feedback"
                  name="feedback"
                  rows={4}
                  value={feedback}
                  onChange={(e) => setFeedback(e.target.value)}
                />
              </div>

              <div className="mb-12">
                <Button type="submit">Submit</Button>
              </div>
            </div>
          </div>
        </div>
      </form>
    </div>
  );
}

export function Radio({ selected, name, value, label, onChange }) {
  return (
    <label className="text-sm font-medium text-gray-700">
      <input
        className="mr-2 shadow-sm sm:text-sm"
        type="radio"
        name={name}
        value={value}
        checked={selected === value}
        onChange={onChange}
      />
      {label}
    </label>
  );
}
